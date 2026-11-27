"""Plain-text library presentation over operation-owned background jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, time, timezone

from here.memory.models import (
    CitationTarget,
    MeetingDetail,
    MeetingPatch,
    MeetingQuery,
    MeetingStatus,
    MeetingSummary,
    SearchHit,
    SearchQuery,
)
from here.memory.search import normalize_query_text
from here.memory.service import Conflict, InvalidInput
from here.ui.bridge import BackgroundJobs
from here.ui.contract import VisualController
from PySide6.QtCore import QDate, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_NO_DATE = QDate(1900, 1, 1)
_AUXILIARY_READS = frozenset({"library-participants", "library-detail"})
_STATUS_LABELS = {
    MeetingStatus.RECORDING: "Grabando",
    MeetingStatus.PROCESSING: "Procesando",
    MeetingStatus.COMPLETED: "Completada",
    MeetingStatus.FAILED: "Fallida",
    MeetingStatus.CANCELLED: "Cancelada",
    MeetingStatus.INTERRUPTED: "Interrumpida",
}


def _label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def _duration(value):
    if value is None:
        return "Duración desconocida"
    seconds = value // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _date(value):
    if value is None:
        return "Fecha desconocida"
    return datetime.fromisoformat(value).astimezone().strftime("%d/%m/%Y %H:%M")


def _timing(segment):
    if segment.start_ms is None and segment.end_ms is None:
        return "Sin tiempo medido"
    start = _duration(segment.start_ms) if segment.start_ms is not None else "Inicio desconocido"
    end = _duration(segment.end_ms) if segment.end_ms is not None else "Fin desconocido"
    return f"{start} – {end}"


class _DateFilter(QDateEdit):
    def focusNextPrevChild(self, forward):  # noqa: N802
        # Date sections are not separate stops in the library's control order.
        # The unset special value has no section to advance to.
        return QWidget.focusNextPrevChild(self, forward)


class LibraryView(QWidget):
    """Render canonical pages; stale requests cannot mutate the current view."""

    meeting_selected = Signal(str)
    segment_selected = Signal(object)
    recording_requested = Signal()
    _closing = Signal()

    def __init__(self, controller: VisualController, jobs: BackgroundJobs, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._jobs = jobs
        self._closed = False
        self._started = False
        self._generation = 0
        self._detail_generation = 0
        self._participant_generation = 0
        self._requests: dict[str, tuple[int, int]] = {}
        self._deletion_jobs: set[int] = set()
        self._read_workers: dict[int, str] = {}
        self._pending_reads: dict[str, tuple[Callable[[], object], int]] = {}
        self._cursor = None
        self._query = MeetingQuery()
        self._items: list[MeetingSummary | SearchHit] = []
        self._detail: MeetingDetail | None = None
        self._detail_notice = ""
        self._speaker_names: dict[tuple[str, str], str] = {}
        self._loading = False

        self._search = QLineEdit()
        self._search.setAccessibleName("Buscar en la biblioteca")
        self._search.setPlaceholderText("Buscar reuniones y transcripciones")
        self._from = self._date_control("Desde: fecha de inicio")
        self._to = self._date_control("Hasta: fecha de fin exclusiva")
        self._status = QComboBox()
        self._status.setAccessibleName("Filtrar por estado")
        self._status.addItem("Todos los estados", None)
        for status, title in _STATUS_LABELS.items():
            self._status.addItem(title, status)
        self._participant = QComboBox()
        self._participant.setAccessibleName("Filtrar por participante")
        self._participant.addItem("Todos los participantes", None)
        self._partial = QCheckBox("Incluir transcripciones parciales")
        self._partial.setAccessibleName("Incluir transcripciones parciales")
        filters = QHBoxLayout()
        for title, control in (("Desde", self._from), ("Hasta (exclusiva)", self._to)):
            label = _label(title)
            label.setBuddy(control)
            filters.addWidget(label)
            filters.addWidget(control)
        filters.addWidget(self._status)
        filters.addWidget(self._participant)

        self._message = _label()
        self._warning = _label()
        self._participant_warning = _label()
        self._participant_retry = QPushButton("Reintentar participantes")
        self._participant_retry.setAccessibleName("Reintentar carga de participantes")
        self._participant_retry.clicked.connect(self._load_participants)
        self._participant_retry.hide()
        self._record = QPushButton("Iniciar grabación")
        self._record.setAccessibleName("Iniciar grabación")
        self._record.clicked.connect(self.recording_requested)
        self._clear = QPushButton("Limpiar filtros")
        self._clear.setAccessibleName("Limpiar filtros de búsqueda")
        self._clear.clicked.connect(self.clear_filters)
        self._retry = QPushButton("Reintentar consulta")
        self._retry.setAccessibleName("Reintentar consulta de biblioteca")
        self._retry.clicked.connect(self.refresh)

        self._results = QTableWidget(0, 6)
        self._results.setAccessibleName("Resultados de reuniones de biblioteca")
        self._results.setHorizontalHeaderLabels(
            ["Título / coincidencia", "Fecha", "Duración", "Estado", "Audio", "Transcripción"]
        )
        self._results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._results.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._results.setTabKeyNavigation(False)
        self._results.horizontalHeader().setStretchLastSection(True)
        self._results.cellActivated.connect(self._open_row)
        self._more = QPushButton("Cargar más")
        self._more.setAccessibleName("Cargar más resultados")
        self._more.clicked.connect(self._continue)

        self._detail_panel = QWidget()
        detail_layout = QVBoxLayout(self._detail_panel)
        self._detail_message = _label()
        self._participants = _label()
        self._transcript = QListWidget()
        self._transcript.setAccessibleName("Transcripción original por segmentos")
        self._transcript.itemActivated.connect(self._select_segment)
        self._title = QLineEdit()
        self._title.setAccessibleName("Editar título de reunión")
        self._save = QPushButton("Guardar título")
        self._save.setAccessibleName("Guardar título de reunión")
        self._save.clicked.connect(self._save_title)
        self._close = QPushButton("Cerrar detalle")
        self._close.setAccessibleName("Cerrar detalle de reunión")
        self._close.clicked.connect(self.close_detail)
        detail_layout.addWidget(self._detail_message)
        detail_layout.addWidget(self._participants)
        detail_layout.addWidget(self._transcript)
        detail_layout.addWidget(self._title)
        detail_layout.addWidget(self._save)
        detail_layout.addWidget(_label("Controles de reproducción pendientes"))
        detail_layout.addWidget(self._close)
        self._detail_panel.hide()

        layout = QVBoxLayout(self)
        layout.addWidget(self._search)
        layout.addLayout(filters)
        layout.addWidget(self._partial)
        layout.addWidget(self._warning)
        layout.addWidget(self._participant_warning)
        layout.addWidget(self._participant_retry)
        layout.addWidget(self._message)
        for control in (self._record, self._clear, self._retry):
            layout.addWidget(control)
            control.hide()
        layout.addWidget(self._results)
        layout.addWidget(self._more)
        layout.addWidget(self._detail_panel)
        self._more.hide()
        order = (
            self._search,
            self._from,
            self._to,
            self._status,
            self._participant,
            self._partial,
            self._results,
            self._transcript,
            self._title,
            self._save,
            self._close,
            self._more,
        )
        for previous, following in zip(order, order[1:]):
            QWidget.setTabOrder(previous, following)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(200)
        self._debounce.timeout.connect(self._request_first)
        self._search.textChanged.connect(self._query_changed)
        self._from.dateChanged.connect(self._query_changed)
        self._to.dateChanged.connect(self._query_changed)
        self._status.currentIndexChanged.connect(self._query_changed)
        self._participant.currentIndexChanged.connect(self._query_changed)
        self._partial.toggled.connect(self._query_changed)
        self._jobs.finished.connect(self._finished)
        self._jobs.settled.connect(self._read_settled)

    @staticmethod
    def _date_control(name):
        control = _DateFilter()
        control.setAccessibleName(name)
        control.setMinimumDate(_NO_DATE)
        control.setSpecialValueText("Sin límite")
        control.setDate(_NO_DATE)
        control.setCalendarPopup(True)
        return control

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not self._started:
            self._started = True
            self._load_participants()
            self.refresh()

    def _submit(self, kind, operation, generation):
        if self._closed or self._deletion_jobs:
            return
        if kind == "library-page" and not self._auxiliary_running():
            # Preserve latest-page responsiveness while an older PAGE is running.
            # This intentionally does not serialize page/page backend operations.
            self._pending_reads.pop(kind, None)
            self._start_read(kind, operation, generation)
        elif kind == "library-page" or kind in _AUXILIARY_READS:
            self._pending_reads[kind] = (operation, generation)
            self._dispatch_pending_reads()
        else:
            self._start_job(kind, operation, generation)

    def _start_job(self, kind, operation, generation):
        request = self._jobs.submit(kind, operation)
        if request is not None:
            self._requests[kind] = (request, generation)
        return request

    def _start_read(self, kind, operation, generation):
        request = self._start_job(kind, operation, generation)
        if request is not None:
            self._read_workers[request] = kind

    def _auxiliary_running(self):
        return any(kind in _AUXILIARY_READS for kind in self._read_workers.values())

    def _read_generation(self, kind):
        if kind == "library-participants":
            return self._participant_generation
        if kind == "library-detail":
            return self._detail_generation
        return self._generation

    def _dispatch_pending_reads(self):
        if self._closed or self._deletion_jobs or self._auxiliary_running():
            return
        page = self._pending_reads.pop("library-page", None)
        if page is not None:
            operation, generation = page
            if generation == self._generation:
                self._start_read("library-page", operation, generation)
                return
        if self._read_workers:
            return
        while self._pending_reads:
            kind = next(iter(self._pending_reads))
            operation, generation = self._pending_reads.pop(kind)
            if generation == self._read_generation(kind):
                self._start_read(kind, operation, generation)
                return

    def _read_settled(self, request):
        was_read = self._read_workers.pop(request, None) is not None
        if request in self._deletion_jobs:
            self._deletion_jobs.remove(request)
            if not self._deletion_jobs:
                # The ACK precedes B's durable barrier. Only the worker's
                # actual exit permits replacement reads.
                self.invalidate_catalog()
        elif was_read:
            self._dispatch_pending_reads()

    def _load_participants(self):
        if self._closed or self._deletion_jobs:
            return
        self._participant_generation += 1
        self._participant_retry.setEnabled(False)
        self._submit(
            "library-participants", self._controller.list_participants, self._participant_generation
        )

    def _query_changed(self, *_):
        if self._closed or self._deletion_jobs:
            return
        # Invalidate immediately, including the 200 ms before a replacement is sent.
        self._generation += 1
        self._cursor = None
        self._loading = False
        self._more.hide()
        self.close_detail()
        self._debounce.start()

    @staticmethod
    def _bound(control):
        if control.date() == _NO_DATE:
            return None
        local = datetime.combine(control.date().toPython(), time.min).astimezone()
        return local.astimezone(timezone.utc).isoformat()

    def _make_query(self):
        text = normalize_query_text(self._search.text())
        query_type = SearchQuery if text else MeetingQuery
        status = self._status.currentData()
        return query_type(
            text=text,
            from_utc=self._bound(self._from),
            to_utc=self._bound(self._to),
            statuses=(MeetingStatus(status),) if status is not None else (),
            participant_id=self._participant.currentData(),
            include_partial=self._partial.isChecked(),
        )

    def clear_filters(self):
        controls = (
            self._search,
            self._from,
            self._to,
            self._status,
            self._participant,
            self._partial,
        )
        for control in controls:
            control.blockSignals(True)
        self._search.clear()
        self._from.setDate(_NO_DATE)
        self._to.setDate(_NO_DATE)
        self._status.setCurrentIndex(0)
        self._participant.setCurrentIndex(0)
        self._partial.setChecked(False)
        for control in controls:
            control.blockSignals(False)
        self.refresh()

    def refresh(self):
        if self._closed or self._deletion_jobs:
            return
        self._debounce.stop()
        self._generation += 1
        self.close_detail()
        self._request_first()

    def _request_first(self):
        if self._closed or self._deletion_jobs:
            return
        self._query = self._make_query()
        self._cursor = None
        self._items.clear()
        self._results.setRowCount(0)
        self._request_page(self._query)

    def _continue(self):
        if self._cursor is not None and not self._loading and not self._closed:
            self._request_page(replace(self._query, cursor=self._cursor))

    def _request_page(self, query):
        self._loading = True
        self._message.setText("Cargando biblioteca…")
        self._warning.clear()
        self._record.hide()
        self._clear.hide()
        self._retry.hide()
        self._more.setEnabled(False)
        operation = (
            self._controller.search
            if isinstance(query, SearchQuery)
            else (self._controller.list_meetings)
        )
        self._submit("library-page", lambda: operation(query), self._generation)

    def _finished(self, request, kind, value, error):
        if self._closed or kind not in self._requests:
            return
        expected_request, generation = self._requests[kind]
        if kind == "library-participants":
            current = self._participant_generation
        elif kind in {"library-detail", "library-title"}:
            current = self._detail_generation
        else:
            current = self._generation
        if request != expected_request or generation != current:
            return
        del self._requests[kind]
        if kind == "library-participants":
            self._participant_retry.setEnabled(True)
            if error is not None:
                self._participant_warning.setText(str(error))
                self._participant_retry.show()
            else:
                self._participant_warning.clear()
                self._participant_retry.hide()
                selected = self._participant.currentData()
                options = {participant.id: participant.display_name for participant in value}
                self._participant.blockSignals(True)
                self._participant.clear()
                self._participant.addItem("Todos los participantes", None)
                for identity, name in options.items():
                    self._participant.addItem(name, identity)
                index = self._participant.findData(selected)
                self._participant.setCurrentIndex(index if index >= 0 else 0)
                self._participant.blockSignals(False)
                if self._participant.currentData() != selected:
                    self._query_changed()
        elif kind == "library-page":
            self._loading = False
            if error is not None:
                if (
                    isinstance(error, InvalidInput)
                    and error.reason == "corpus_changed"
                    and self._cursor is not None
                ):
                    self._generation += 1
                    self._request_first()
                    return
                self._message.setText(str(error))
                self._retry.show()
                self._more.setEnabled(True)
                return
            self._render_page(value)
        elif kind == "library-detail":
            if error is not None:
                self._detail_message.setText(str(error))
            elif value is None:
                self._detail_message.setText("La reunión ya no está disponible.")
            else:
                self._render_detail(value)
        elif kind == "library-title":
            self._save.setEnabled(True)
            if isinstance(error, Conflict):
                detail = self._detail
                if detail is None:
                    return
                identity = detail.meeting.id
                self._open_detail(identity)
                self._detail_notice = "El título cambió. Revisá la versión actual antes de guardar."
                self._detail_message.setText(self._detail_notice)
                # Refresh rows too; never retry the mutation with a newer version.
                self._generation += 1
                self._request_first()
            elif error is not None:
                self._detail_message.setText(str(error))
            else:
                self.refresh()

    def _render_page(self, page):
        self._items.extend(page.items)
        self._cursor = page.next_cursor
        self._results.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            if isinstance(item, SearchHit):
                values = (
                    f"{item.title}\n{item.snippet}\n{_timing(item)}",
                    _date(item.started_at),
                    "Duración desconocida",
                    "",
                    "Disponibilidad no consultada",
                    "Parcial" if item.partial else "Completa",
                )
            else:
                meeting = item.meeting
                values = (
                    meeting.title,
                    _date(meeting.started_at),
                    _duration(meeting.duration_ms),
                    _STATUS_LABELS.get(meeting.status, meeting.status.value),
                    "Audio disponible" if item.audio_available else "Audio no disponible",
                    "Sin transcripción"
                    if meeting.active_revision_id is None
                    else ("Parcial" if item.partial else "Completa"),
                )
            for column, text in enumerate(values):
                self._results.setItem(row, column, QTableWidgetItem(text))
        self._results.resizeRowsToContents()
        self._more.setVisible(self._cursor is not None)
        self._more.setEnabled(True)
        self._warning.setText(
            "Búsqueda literal con capacidad limitada (sin FTS5)."
            if getattr(page, "degraded", False)
            else ""
        )
        filtered = self._query != MeetingQuery()
        self._message.setText(
            "" if self._items else ("No hay coincidencias" if filtered else "Aún no hay reuniones")
        )
        self._record.setVisible(not self._items and not filtered)
        self._clear.setVisible(not self._items and filtered)

    def _open_row(self, row, _column=0):
        if 0 <= row < len(self._items):
            item = self._items[row]
            identity = item.meeting_id if isinstance(item, SearchHit) else item.meeting.id
            self.meeting_selected.emit(identity)
            self._open_detail(identity)

    def _open_detail(self, identity):
        self.close_detail()
        self._detail_panel.show()
        self._detail_message.setText("Cargando detalle…")
        self._title.setEnabled(False)
        self._save.setEnabled(False)
        self._submit(
            "library-detail",
            lambda: self._controller.get_meeting(identity),
            self._detail_generation,
        )

    def _render_detail(self, detail):
        self._detail = detail
        self._title.setText(detail.meeting.title)
        self._title.setEnabled(True)
        self._save.setEnabled(True)
        messages = [self._detail_notice] if self._detail_notice else []
        messages.extend(str(warning) for warning in detail.warnings)
        self._detail_message.setText("\n".join(messages))
        names = {participant.id: participant.display_name for participant in detail.participants}
        self._participants.setText(
            "Participantes: " + ", ".join(names.values())
            if names
            else "Sin participantes registrados"
        )
        self._speaker_names = {
            (assignment.revision_id, assignment.speaker_key): names[assignment.participant_id]
            for assignment in detail.speaker_assignments
            if assignment.meeting_id == detail.meeting.id and assignment.participant_id in names
        }
        revisions = {revision.id: revision for revision in detail.revisions}
        for segment in detail.segments:
            revision = revisions.get(segment.revision_id)
            context = (
                "Actual"
                if segment.revision_id == detail.meeting.active_revision_id
                else ("Histórica")
            )
            if revision is not None:
                context += f" · revisión {revision.revision_number}"
                if revision.status == "partial":
                    context += " · Parcial"
            speaker = self._speaker_names.get((segment.revision_id, segment.speaker_key))
            if speaker is not None:
                context += f" · {speaker}"
            item = QListWidgetItem(f"{context} · {_timing(segment)}\n{segment.text}")
            item.setData(Qt.ItemDataRole.UserRole, segment)
            self._transcript.addItem(item)

    def _select_segment(self, item):
        if self._detail is None:
            return
        segment = item.data(Qt.ItemDataRole.UserRole)
        meeting = self._detail.meeting
        self.segment_selected.emit(
            CitationTarget(
                meeting_id=meeting.id,
                revision_id=segment.revision_id,
                segment_id=segment.id,
                title=meeting.title,
                started_at=meeting.started_at,
                text=segment.text,
                speaker=self._speaker_names.get((segment.revision_id, segment.speaker_key)),
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                timing_source=segment.timing_source,
                status="current"
                if segment.revision_id == meeting.active_revision_id
                else "historical",
            )
        )

    def _save_title(self):
        if self._detail is None or self._closed:
            return
        meeting = self._detail.meeting
        patch = MeetingPatch(title=self._title.text())
        self._save.setEnabled(False)
        self._submit(
            "library-title",
            lambda: self._controller.update_meeting(meeting.id, patch, meeting.row_version),
            self._detail_generation,
        )

    def close_detail(self):
        self._detail_generation += 1
        self._detail = None
        self._detail_notice = ""
        self._speaker_names.clear()
        self._participants.clear()
        self._transcript.clear()
        self._title.clear()
        self._detail_panel.hide()

    def invalidate_catalog(self):
        """Refresh accepted catalog changes; no deletion authority is inferred here."""
        if self._started and not self._closed:
            self._load_participants()
            self.refresh()

    def _pause_for_delete(self, request: int) -> None:
        if self._closed:
            raise RuntimeError("Library view is closed")
        self._deletion_jobs.add(request)
        self._debounce.stop()
        self._generation += 1
        self._participant_generation += 1
        self._pending_reads.clear()
        self._requests.clear()
        self.close_detail()
        self._cursor = None
        self._items.clear()
        self._results.setRowCount(0)
        self._loading = False
        self._message.clear()
        self._warning.clear()
        self._participant_warning.clear()
        self._participant_retry.hide()
        self._record.hide()
        self._clear.hide()
        self._retry.hide()
        self._more.hide()

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._closing.emit()
        self._deletion_jobs.clear()
        self._pending_reads.clear()
        self._generation += 1
        self._debounce.stop()
        self.close_detail()
