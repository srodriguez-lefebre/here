/* Private aggregate for the canonical search corpus v2 JSON frame. */
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "sqlite3ext.h"

/* Every connection must use the same SQLite API table, including concurrent loads. */
static _Atomic(const sqlite3_api_routines *) sqlite_api;
#define sqlite3_api atomic_load_explicit(&sqlite_api, memory_order_acquire)

enum {
    FTS_COLUMNS = 5,
    BATCH_ROWS = 256,
    EMPTY_TUPLE_BYTES = FTS_COLUMNS * sizeof(uint32_t),
    ENVELOPE_HEADER_BYTES = 14,
    ENVELOPE_LIMIT = 8 * 1024 * 1024
};

typedef struct {
    unsigned char *bytes;
    size_t length;
    size_t capacity;
} Buffer;

typedef struct {
    const unsigned char *bytes;
    size_t length;
} Text;

typedef struct {
    Buffer first;
    Buffer last;
    Buffer json;
    sqlite3_int64 rowid;
    unsigned int rows;
    int error;
} Batch;

static void write_be(unsigned char *output, size_t width, uint64_t value) {
    while (width > 0) {
        output[--width] = (unsigned char)value;
        value >>= 8;
    }
}

static uint32_t read_length(const unsigned char *input) {
    return ((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) |
           ((uint32_t)input[2] << 8) | input[3];
}

static int reserve(Buffer *buffer, size_t required, size_t limit) {
    if (required > limit) {
        return SQLITE_MISMATCH;
    }
    if (required <= buffer->capacity) {
        return SQLITE_OK;
    }
    size_t capacity = buffer->capacity > 0 ? buffer->capacity : 256;
    while (capacity < required) {
        capacity *= 2;
    }
    if (capacity > limit) {
        capacity = limit;
    }
    unsigned char *bytes = sqlite3_realloc64(buffer->bytes, capacity);
    if (bytes == NULL) {
        return SQLITE_NOMEM;
    }
    buffer->bytes = bytes;
    buffer->capacity = capacity;
    return SQLITE_OK;
}

static int append(Buffer *buffer, const void *bytes, size_t length, size_t limit) {
    if (length == 0) {
        return SQLITE_OK;
    }
    int error = reserve(buffer, buffer->length + length, limit);
    if (error == SQLITE_OK) {
        memcpy(buffer->bytes + buffer->length, bytes, length);
        buffer->length += length;
    }
    return error;
}

/* Skip ASCII words and validate the remaining UTF-8 scalar values. */
static int valid_utf8(const unsigned char *bytes, size_t length) {
    while (length > 0) {
        if (length >= sizeof(uint64_t)) {
            uint64_t word;
            memcpy(&word, bytes, sizeof(word));
            if ((word & UINT64_C(0x8080808080808080)) == 0) {
                bytes += sizeof(word);
                length -= sizeof(word);
                continue;
            }
        }
        unsigned char first = *bytes++;
        --length;
        if (first < 0x80) {
            continue;
        }
        size_t extra;
        unsigned char second_min = 0x80;
        unsigned char second_max = 0xbf;
        if (first >= 0xc2 && first <= 0xdf) {
            extra = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            extra = 2;
            if (first == 0xe0) {
                second_min = 0xa0;
            } else if (first == 0xed) {
                second_max = 0x9f;
            }
        } else if (first >= 0xf0 && first <= 0xf4) {
            extra = 3;
            if (first == 0xf0) {
                second_min = 0x90;
            } else if (first == 0xf4) {
                second_max = 0x8f;
            }
        } else {
            return 0;
        }
        if (length < extra || bytes[0] < second_min || bytes[0] > second_max) {
            return 0;
        }
        for (size_t i = 1; i < extra; ++i) {
            if ((bytes[i] & 0xc0) != 0x80) {
                return 0;
            }
        }
        bytes += extra;
        length -= extra;
    }
    return 1;
}

/* Compare the original UTF-8 bytes under SQLite BINARY, including embedded NUL. */
static int precedes(const Text *cells, const unsigned char *previous) {
    for (int column = 0; column < FTS_COLUMNS; ++column) {
        size_t old_length = read_length(previous);
        previous += sizeof(uint32_t);
        size_t length = cells[column].length;
        size_t shared = length < old_length ? length : old_length;
        int order = memcmp(cells[column].bytes, previous, shared);
        if (order != 0) {
            return order < 0;
        }
        if (length != old_length) {
            return length < old_length;
        }
        previous += old_length;
    }
    return 0;
}

static void write_tuple(unsigned char *output, const Text *cells) {
    for (int column = 0; column < FTS_COLUMNS; ++column) {
        size_t length = cells[column].length;
        write_be(output, sizeof(uint32_t), length);
        output += sizeof(uint32_t);
        memcpy(output, cells[column].bytes, length);
        output += length;
    }
}

static int append_escape(Buffer *buffer, unsigned char byte, size_t limit) {
    static const unsigned char hex[] = "0123456789abcdef";
    unsigned char escaped[] = {'\\', 0};
    switch (byte) {
    case '"':
    case '\\': escaped[1] = byte; break;
    case '\b': escaped[1] = 'b'; break;
    case '\f': escaped[1] = 'f'; break;
    case '\n': escaped[1] = 'n'; break;
    case '\r': escaped[1] = 'r'; break;
    case '\t': escaped[1] = 't'; break;
    default: {
        unsigned char unicode[] = {'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15]};
        return append(buffer, unicode, sizeof(unicode), limit);
    }
    }
    return append(buffer, escaped, sizeof(escaped), limit);
}

/* Match json.dumps(..., ensure_ascii=False, separators=(",", ":")). */
static int append_json_text(Buffer *buffer, Text cell, size_t limit) {
    int error = append(buffer, "\"", 1, limit);
    const unsigned char *run = cell.bytes;
    const unsigned char *end = run + cell.length;
    for (const unsigned char *cursor = run; cursor < end && error == SQLITE_OK; ++cursor) {
        if (*cursor >= 0x20 && *cursor != '"' && *cursor != '\\') {
            continue;
        }
        error = append(buffer, run, (size_t)(cursor - run), limit);
        if (error == SQLITE_OK) {
            error = append_escape(buffer, *cursor, limit);
        }
        run = cursor + 1;
    }
    if (error == SQLITE_OK) {
        error = append(buffer, run, (size_t)(end - run), limit);
    }
    if (error == SQLITE_OK) {
        error = append(buffer, "\"", 1, limit);
    }
    return error;
}

static int append_row(Batch *batch, sqlite3_value **args) {
    if (batch->rows == BATCH_ROWS || sqlite3_value_type(args[0]) != SQLITE_INTEGER) {
        return SQLITE_MISMATCH;
    }
    sqlite3_int64 rowid = sqlite3_value_int64(args[0]);
    if (batch->rows > 0 && rowid <= batch->rowid) {
        return SQLITE_MISMATCH;
    }

    Text cells[FTS_COLUMNS];
    size_t tuple_length = EMPTY_TUPLE_BYTES;
    for (int column = 0; column < FTS_COLUMNS; ++column) {
        sqlite3_value *value = args[column + 1];
        if (sqlite3_value_type(value) != SQLITE_TEXT) {
            return SQLITE_MISMATCH;
        }
        const unsigned char *bytes = sqlite3_value_text(value);
        if (bytes == NULL) {
            return SQLITE_NOMEM;
        }
        size_t length = (size_t)sqlite3_value_bytes(value);
        if (length > ENVELOPE_LIMIT - tuple_length || !valid_utf8(bytes, length)) {
            return SQLITE_MISMATCH;
        }
        cells[column] = (Text){bytes, length};
        tuple_length += length;
    }
    if (batch->rows > 0 && precedes(cells, batch->last.bytes)) {
        return SQLITE_MISMATCH;
    }

    size_t first_length = batch->rows == 0 ? tuple_length : batch->first.length;
    /* The final last tuple may be shorter than an intermediate one. */
    size_t overhead = ENVELOPE_HEADER_BYTES + first_length + EMPTY_TUPLE_BYTES;
    if (overhead >= ENVELOPE_LIMIT) {
        return SQLITE_MISMATCH;
    }
    /* Leave room for the closing bracket of the batch. */
    size_t json_limit = ENVELOPE_LIMIT - overhead - 1;
    int error = append(&batch->json, batch->rows == 0 ? "[[" : ",[", 2, json_limit);
    for (int column = 0; column < FTS_COLUMNS && error == SQLITE_OK; ++column) {
        if (column > 0) {
            error = append(&batch->json, ",", 1, json_limit);
        }
        if (error == SQLITE_OK) {
            error = append_json_text(&batch->json, cells[column], json_limit);
        }
    }
    if (error == SQLITE_OK) {
        error = append(&batch->json, "]", 1, json_limit);
    }
    if (error == SQLITE_OK) {
        error = reserve(&batch->last, tuple_length, ENVELOPE_LIMIT);
    }
    if (error != SQLITE_OK) {
        return error;
    }
    write_tuple(batch->last.bytes, cells);
    batch->last.length = tuple_length;
    if (batch->rows == 0) {
        error = append(&batch->first, batch->last.bytes, tuple_length, ENVELOPE_LIMIT);
        if (error != SQLITE_OK) {
            return error;
        }
    }
    batch->rowid = rowid;
    ++batch->rows;
    return SQLITE_OK;
}

static void report_error(sqlite3_context *context, int error) {
    if (error == SQLITE_MISMATCH) {
        sqlite3_result_error(context, "here search v2: FTS input unsupported", -1);
        sqlite3_result_error_code(context, error);
    } else {
        sqlite3_result_error_nomem(context);
    }
}

static void batch_step(sqlite3_context *context, int argc, sqlite3_value **args) {
    (void)argc;
    Batch *batch = sqlite3_aggregate_context(context, sizeof(*batch));
    if (batch == NULL) {
        sqlite3_result_error_nomem(context);
        return;
    }
    if (batch->error == SQLITE_OK) {
        batch->error = append_row(batch, args);
    }
    if (batch->error != SQLITE_OK) {
        report_error(context, batch->error);
    }
}

static void result_frame(sqlite3_context *context, const Batch *batch) {
    size_t length = ENVELOPE_HEADER_BYTES + batch->first.length + batch->last.length +
                    batch->json.length + 1;
    if (length > ENVELOPE_LIMIT) {
        report_error(context, SQLITE_MISMATCH);
        return;
    }
    unsigned char *frame = sqlite3_malloc64(length);
    if (frame == NULL) {
        sqlite3_result_error_nomem(context);
        return;
    }
    memcpy(frame, "HV2F", 4);
    write_be(frame + 4, 2, batch->rows);
    write_be(frame + 6, 8, (uint64_t)batch->rowid);

    unsigned char *output = frame + ENVELOPE_HEADER_BYTES;
    memcpy(output, batch->first.bytes, batch->first.length);
    output += batch->first.length;
    memcpy(output, batch->last.bytes, batch->last.length);
    output += batch->last.length;
    memcpy(output, batch->json.bytes, batch->json.length);
    output[batch->json.length] = ']';
    sqlite3_result_blob64(context, frame, length, sqlite3_free);
}

static void batch_final(sqlite3_context *context) {
    Batch *batch = sqlite3_aggregate_context(context, 0);
    if (batch == NULL) {
        sqlite3_result_null(context);
        return;
    }
    if (batch->error != SQLITE_OK) {
        report_error(context, batch->error);
    } else if (batch->rows > 0) {
        result_frame(context, batch);
    } else {
        sqlite3_result_null(context);
    }
    sqlite3_free(batch->first.bytes);
    sqlite3_free(batch->last.bytes);
    sqlite3_free(batch->json.bytes);
}

__declspec(dllexport) int sqlite3_here_search_v2_init(
    sqlite3 *db, char **error_message, const sqlite3_api_routines *api) {
    (void)error_message;
    if (api == NULL || api->sourceid == NULL || strcmp(api->sourceid(), SQLITE_SOURCE_ID) != 0) {
        return SQLITE_MISMATCH;
    }
    const sqlite3_api_routines *expected = NULL;
    if (!atomic_compare_exchange_strong_explicit(
            &sqlite_api, &expected, api, memory_order_acq_rel, memory_order_acquire) &&
        expected != api) {
        return SQLITE_MISUSE;
    }
    return sqlite3_create_function_v2(
        db, "here_search_v2_fts_batch", FTS_COLUMNS + 1, SQLITE_UTF8 | SQLITE_DIRECTONLY,
        NULL, NULL, batch_step, batch_final, NULL);
}
