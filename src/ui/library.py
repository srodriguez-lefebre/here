"""Plain-text library presentation over operation-owned background jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, time, timezone
from pathlib import Path
from threading import Event, Lock

from here.memory.database import IntegrityFailure, UnsupportedSchema
from here.memory.models import (
    Citation,
    CitationTarget,
    MeetingDetail,
    MeetingPatch,
    MeetingQuery,
    MeetingStatus,
    MeetingSummary,
    SearchHit,
    SearchQuery,
)
from here.memory.portability import (
    ExportCancelled,
    ExportResult,
    ImportCancelled,
    ImportReport,
    InsufficientSpace,
    InvalidPackage,
)
from here.memory.search import normalize_query_text
from here.memory.service import Conflict, InvalidInput, NotFound
from here.ui.bridge import BackgroundJobs
from here.ui.contract import ApplicationState, VisualController
from here.ui.playback_widget import PlaybackWidget
from PySide6.QtCore import QDate, QItemSelectionModel, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
_DETAIL_READS = frozenset({"library-detail", "library-citation"})
_AUXILIARY_READS = _DETAIL_READS | {"library-participants"}
_EXPORT_PACKAGE_FILTER = "Paquetes de here (*.here.zip)"
_EXPORT_MARKDOWN_FILTER = "Documentos Markdown (*.md)"
_IMPORT_PACKAGE_FILTER = "Paquetes de here (*.zip)"
_IMPORT_STATUS = {
    "inserted": "Importada",
    "unchanged": "Sin cambios",
    "conflict": "Conflicto",
    "rejected": "Rechazada",
}
_IMPORT_REASON = {
    "local_io": "Falló una operación local de almacenamiento",
    "local_recovery_pending": "Recuperación local pendiente",
    "batch_stopped": "Se detuvo el lote",
    "suppressed": "Reunión suprimida",
    "different_meeting": "El identificador corresponde a otra reunión",
    "audio_unavailable": "Audio local no disponible",
    "destination_exists": "Ya existe el destino local",
    "participant_exists": "Ya existe un participante con ese identificador",
    "catalog_identity_exists": "Ya existe un registro con ese identificador",
}
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


class _PortableProgress:
    """Share only the latest measured byte count with the Qt thread."""

    def __init__(self):
        self.cancelled = Event()
        self._lock = Lock()
        self._bytes: int | None = None

    def report(self, byte_count: int) -> None:
        with self._lock:
            self._bytes = byte_count

    def latest(self) -> int | None:
        with self._lock:
            return self._bytes


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
        self._citation_generation = 0
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
        self._restore_selection: frozenset[str] = frozenset()
        self._portable_job: tuple[int, _PortableProgress] | None = None

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
        self._results.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._results.setTabKeyNavigation(False)
        self._results.horizontalHeader().setStretchLastSection(True)
        self._results.cellActivated.connect(self._open_row)
        self._more = QPushButton("Cargar más")
        self._more.setAccessibleName("Cargar más resultados")
        self._more.clicked.connect(self._continue)

        self._export = QPushButton("Exportar selección")
        self._export.clicked.connect(self._choose_export)
        self._import = QPushButton("Importar paquete")
        self._import.clicked.connect(self._choose_import)
        self._portable_cancel = QPushButton("Cancelar transferencia")
        self._portable_cancel.clicked.connect(self._cancel_portable_job)
        self._portable_cancel.hide()
        self._portable_progress = _label()
        self._portable_report = QListWidget()
        self._portable_report.setAccessibleName("Resultado por reunión")
        self._portable_report.setMaximumHeight(120)
        self._portable_report.hide()
        self._portable_timer = QTimer(self)
        self._portable_timer.setInterval(50)
        self._portable_timer.timeout.connect(self._show_portable_progress)

        self._detail_panel = QWidget()
        detail_layout = QVBoxLayout(self._detail_panel)
        self._detail_message = _label()
        self._citation_message = _label()
        self._citation_message.hide()
        self._participants = _label()
        self._transcript = QListWidget()
        self._transcript.setAccessibleName("Transcripción original por segmentos")
        self._transcript.itemActivated.connect(self._select_segment)
        self._revision_badge = _label()
        self._revision_badge.hide()
        self._current_revision = QPushButton("Abrir revisión actual")
        self._current_revision.clicked.connect(self._open_current_revision)
        self._current_revision.hide()
        self._playback = PlaybackWidget(self._detail_panel)
        self._playback.hide()
        self._title = QLineEdit()
        self._title.setAccessibleName("Editar título de reunión")
        self._save = QPushButton("Guardar título")
        self._save.setAccessibleName("Guardar título de reunión")
        self._save.clicked.connect(self._save_title)
        self._export_markdown = QPushButton("Exportar Markdown")
        self._export_markdown.clicked.connect(self._choose_markdown_export)
        self._close = QPushButton("Cerrar detalle")
        self._close.setAccessibleName("Cerrar detalle de reunión")
        self._close.clicked.connect(self.close_detail)
        detail_layout.addWidget(self._detail_message)
        detail_layout.addWidget(self._participants)
        detail_layout.addWidget(self._transcript)
        detail_layout.addWidget(self._title)
        detail_layout.addWidget(self._save)
        detail_layout.addWidget(self._citation_message)
        detail_layout.addWidget(self._revision_badge)
        detail_layout.addWidget(self._current_revision)
        detail_layout.addWidget(self._playback)
        detail_layout.addWidget(self._export_markdown)
        detail_layout.addWidget(self._close)
        self._detail_panel.hide()
        self._results.itemSelectionChanged.connect(self._update_portable_actions)
        self._update_portable_actions()

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
        transfer_actions = QHBoxLayout()
        for button in (self._export, self._import, self._portable_cancel):
            transfer_actions.addWidget(button)
        transfer_actions.addStretch()
        layout.addLayout(transfer_actions)
        layout.addWidget(self._portable_progress)
        layout.addWidget(self._portable_report)
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
            self._export_markdown,
            self._close,
            self._more,
            self._export,
            self._import,
            self._portable_cancel,
            self._portable_report,
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
        if kind == "library-citation":
            return self._citation_generation
        if kind in {"library-detail", "library-title"}:
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
                self._load_participants()
                self._request_first()
            self._update_portable_actions()
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

    def _request_first(self, *, retain_selection=False):
        if self._closed or self._deletion_jobs:
            return
        self._restore_selection = (
            frozenset(self._selected_meeting_ids()) if retain_selection else frozenset()
        )
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
        if kind in {"portable-export", "portable-import", "portable-markdown"}:
            self._finish_portable_job(request, kind, value, error)
            return
        if self._closed or kind not in self._requests:
            return
        expected_request, generation = self._requests[kind]
        if request != expected_request or generation != self._read_generation(kind):
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
        elif kind == "library-citation":
            if error is not None:
                self._citation_message.setText(str(error))
                self._citation_message.show()
            else:
                self._render_target(value)
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
        restore_selection, self._restore_selection = self._restore_selection, frozenset()
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
            if self._meeting_id(item) in restore_selection:
                model = self._results.selectionModel()
                index = self._results.model().index(row, 0)
                flags = QItemSelectionModel.SelectionFlag
                model.select(index, flags.Select | flags.Rows)
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

    def _selected_meeting_ids(self) -> tuple[str, ...]:
        rows = sorted(
            index.row()
            for index in self._results.selectionModel().selectedRows()
            if 0 <= index.row() < len(self._items)
        )
        return tuple(dict.fromkeys(self._meeting_id(self._items[row]) for row in rows))

    def _can_transfer(self) -> bool:
        return not self._closed and not self._deletion_jobs and self._portable_job is None

    def _update_portable_actions(self):
        enabled = self._can_transfer()
        self._export.setEnabled(enabled and bool(self._selected_meeting_ids()))
        self._import.setEnabled(enabled)
        self._export_markdown.setEnabled(enabled and self._detail is not None)

    def _choose_export(self):
        ids = self._selected_meeting_ids()
        if not ids or not self._can_transfer():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Exportar reuniones")
        checkbox = QCheckBox("Incluir audio")
        checkbox.setChecked(True)
        options = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        options.accepted.connect(dialog.accept)
        options.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(checkbox)
        layout.addWidget(options)
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted.value
            include_audio = checkbox.isChecked()
        finally:
            dialog.deleteLater()
        if not accepted or not self._can_transfer():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar reuniones", "reuniones.here.zip", _EXPORT_PACKAGE_FILTER
        )
        if path:
            destination = Path(path)
            export = self._controller.export_meetings
            self._start_portable_job(
                "portable-export",
                "Preparando exportación…",
                lambda report, cancelled: export(
                    ids, destination, include_audio, report, cancelled
                ),
            )

    def _choose_markdown_export(self):
        if self._detail is None or not self._can_transfer():
            return
        meeting_id = self._detail.meeting.id
        generation = self._detail_generation
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Markdown", "reunión.md", _EXPORT_MARKDOWN_FILTER
        )
        if not path or generation != self._detail_generation:
            return
        destination = Path(path)
        export = self._controller.export_meeting_markdown
        self._start_portable_job(
            "portable-markdown",
            "Exportando Markdown…",
            lambda _report, cancelled: export(meeting_id, destination, cancelled),
        )

    def _choose_import(self):
        if not self._can_transfer():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Importar paquete", "", _IMPORT_PACKAGE_FILTER)
        if path:
            source = Path(path)
            import_archive = self._controller.import_meetings
            self._start_portable_job(
                "portable-import",
                "Preparando importación…",
                lambda report, cancelled: import_archive(source, report, cancelled),
            )

    def _start_portable_job(
        self,
        kind: str,
        message: str,
        operation: Callable[
            [Callable[[int], None], Callable[[], bool]], ExportResult | ImportReport | Path
        ],
    ):
        if not self._can_transfer():
            return
        progress = _PortableProgress()
        request = self._jobs.submit(
            kind, lambda: operation(progress.report, progress.cancelled.is_set)
        )
        if request is None:
            return
        self._portable_job = (request, progress)
        self._portable_report.clear()
        self._portable_report.hide()
        self._portable_progress.setText(message)
        self._portable_cancel.setEnabled(True)
        self._portable_cancel.show()
        self._update_portable_actions()
        self._portable_timer.start()

    def _show_portable_progress(self):
        if self._portable_job is None:
            return
        _, progress = self._portable_job
        byte_count = progress.latest()
        cancelling = progress.cancelled.is_set()
        if byte_count is not None:
            text = f"{byte_count} bytes procesados"
            self._portable_progress.setText(f"Cancelando… {text}" if cancelling else text)
        elif cancelling:
            self._portable_progress.setText("Cancelando…")

    def _cancel_portable_job(self):
        if self._portable_job is not None:
            _, progress = self._portable_job
            progress.cancelled.set()
            self._portable_cancel.setEnabled(False)
            self._show_portable_progress()

    def _finish_portable_job(self, request, kind, value, error):
        if self._portable_job is None or self._closed or request != self._portable_job[0]:
            return
        self._portable_timer.stop()
        self._portable_job = None
        self._portable_cancel.hide()
        self._update_portable_actions()
        if error is not None:
            message = self._portable_error(kind, error)
        elif kind == "portable-markdown":
            message = "Exportación Markdown finalizada."
        elif isinstance(value, ExportResult):
            message = "\n".join((f"Exportación finalizada · {value.path}", *value.warnings))
        else:
            message = "Importación finalizada"
            self._show_import_report(value)
        self._portable_progress.setText(message)

    @staticmethod
    def _portable_error(kind, error):
        if isinstance(error, (ExportCancelled, ImportCancelled)):
            return "Transferencia cancelada."
        if isinstance(error, InsufficientSpace):
            if kind == "portable-markdown":
                return "No hay espacio suficiente para guardar el documento."
            return "No hay espacio suficiente para guardar el paquete."
        if isinstance(error, InvalidPackage):
            return "El paquete es inválido o no se pudo leer."
        if isinstance(error, UnsupportedSchema):
            return "La versión del paquete o del catálogo no es compatible."
        if isinstance(error, IntegrityFailure):
            return "Falló una comprobación de integridad."
        if kind == "portable-markdown":
            if isinstance(error, NotFound):
                return "La reunión ya no está disponible."
            if isinstance(error, Conflict):
                return "La reunión cambió. Volvé a intentar la exportación."
            return "No se pudo exportar el documento Markdown."
        if isinstance(error, Conflict):
            return "La reunión cambió. Revisá la selección y volvé a intentar."
        if kind == "portable-export":
            return "No se pudo exportar la selección."
        return "No se pudo importar el paquete."

    def _show_import_report(self, report: ImportReport):
        rows: dict[str, str] = {}
        for meeting in report.meetings:
            status = _IMPORT_STATUS[meeting.status]
            reason = _IMPORT_REASON.get(meeting.reason, meeting.reason)
            summary = f"{meeting.meeting_id} · {status}"
            rows[meeting.meeting_id] = f"{summary} · {reason}" if reason else summary
        for identity in report.pending_recovery_ids:
            rows[identity] = f"{identity} · Confirmación pendiente de recuperación local"
        order = report.manifest_order or tuple(rows)
        self._portable_report.addItems([rows[identity] for identity in order if identity in rows])
        if report.cleanup_pending:
            self._portable_report.addItem(
                "Limpieza del área temporal pendiente de recuperación local"
            )
        self._portable_report.setVisible(
            bool(report.meetings or report.pending_recovery_ids or report.cleanup_pending)
        )
        if any(meeting.status == "inserted" for meeting in report.meetings):
            # A cancellation or local recovery warning can follow a durable commit.
            # Refresh the rows without reopening an unchanged detail or playback.
            if not self._deletion_jobs:
                self._debounce.stop()
                self._generation += 1
                self._load_participants()
                self._request_first(retain_selection=True)

    @staticmethod
    def _meeting_id(item: MeetingSummary | SearchHit) -> str:
        return item.meeting_id if isinstance(item, SearchHit) else item.meeting.id

    def _open_row(self, row, _column=0):
        if 0 <= row < len(self._items):
            item = self._items[row]
            identity = self._meeting_id(item)
            self.meeting_selected.emit(identity)
            if isinstance(item, SearchHit):
                self.open_target(Citation(item.meeting_id, item.revision_id, item.segment_id))
            else:
                self._open_detail(identity)

    def set_application_state(self, state: ApplicationState) -> None:
        self._playback.set_application_state(state)

    def open_target(self, target: CitationTarget | Citation) -> None:
        """Open evidence by identity and resolve its authoritative metadata on a worker."""
        if self._closed or self._deletion_jobs:
            return
        citation = (
            target
            if isinstance(target, Citation)
            else Citation(target.meeting_id, target.revision_id, target.segment_id)
        )
        if self._detail is not None and self._detail.meeting.id == citation.meeting_id:
            self._reset_target()
        else:
            self._open_detail(citation.meeting_id)
        self._submit(
            "library-citation",
            lambda: self._controller.resolve_citation(citation),
            self._citation_generation,
        )

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
        self._update_portable_actions()
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
        self._render_transcript(detail)

    def _render_transcript(self, detail: MeetingDetail) -> None:
        self._transcript.clear()
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

    def _render_target(self, target: CitationTarget) -> None:
        detail = self._detail
        if detail is None or detail.meeting.id != target.meeting_id:
            return
        revision_available = any(revision.id == target.revision_id for revision in detail.revisions)
        selected = None
        for index in range(self._transcript.count()):
            item = self._transcript.item(index)
            segment = item.data(Qt.ItemDataRole.UserRole)
            if segment.id == target.segment_id and segment.revision_id == target.revision_id:
                selected = item
                break
        if selected is None and target.status != "missing":
            self._citation_message.setText("Volvé a abrir la reunión para ver este segmento.")
            self._citation_message.show()
            return

        active_revision = detail.meeting.active_revision_id
        if target.status == "historical" or (
            target.status == "missing"
            and selected is not None
            and target.revision_id != active_revision
        ):
            self._revision_badge.setText("Revisión histórica")
            self._revision_badge.show()
            self._current_revision.setVisible(active_revision is not None)
        self._playback.show()
        self._playback.load(target, revision_available=revision_available)
        if selected is not None:
            self._transcript.setCurrentItem(selected)
            self._transcript.scrollToItem(selected, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _open_current_revision(self) -> None:
        detail = self._detail
        if detail is None or detail.meeting.active_revision_id is None:
            return
        revision_id = detail.meeting.active_revision_id
        segment = next((item for item in detail.segments if item.revision_id == revision_id), None)
        if segment is None:
            self._transcript.setCurrentRow(-1)
            self._reset_target()
        else:
            self.open_target(Citation(detail.meeting.id, revision_id, segment.id))

    def _select_segment(self, item):
        if self._detail is None:
            return
        segment = item.data(Qt.ItemDataRole.UserRole)
        meeting = self._detail.meeting
        target = CitationTarget(
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
            status="current" if segment.revision_id == meeting.active_revision_id else "historical",
        )
        self.segment_selected.emit(target)
        self.open_target(target)

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
        self._clear_detail()

    def _clear_detail(self, *, release_playback: bool = True) -> None:
        self._detail_generation += 1
        for kind in _DETAIL_READS:
            self._pending_reads.pop(kind, None)
            self._requests.pop(kind, None)
        self._reset_target(release_playback=release_playback)
        self._detail = None
        self._detail_notice = ""
        self._speaker_names.clear()
        self._participants.clear()
        self._transcript.clear()
        self._title.clear()
        self._detail_panel.hide()
        self._update_portable_actions()

    def _reset_target(self, *, release_playback: bool = True) -> None:
        self._citation_generation += 1
        self._citation_message.clear()
        self._citation_message.hide()
        if release_playback:
            self._playback.clear()
        self._playback.hide()
        self._revision_badge.hide()
        self._current_revision.hide()

    def invalidate_catalog(self):
        """Refresh accepted catalog changes; no deletion authority is inferred here."""
        if self._started and not self._closed:
            self._load_participants()
            self.refresh()

    def _pause_for_delete(self, request: int, meeting_id: str) -> None:
        if self._closed:
            raise RuntimeError("Library view is closed")
        self._deletion_jobs.add(request)
        self._update_portable_actions()
        self._debounce.stop()
        self._generation += 1
        self._participant_generation += 1
        self._pending_reads.clear()
        self._requests.clear()
        media_meeting = self._playback.meeting_id
        if media_meeting is not None and media_meeting != meeting_id:
            if self._detail is None or self._detail.meeting.id != media_meeting:
                self._clear_detail(release_playback=False)
            else:
                self._detail_generation += 1
                self._save.setEnabled(True)
        else:
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
        self._cancel_portable_job()
        self._portable_timer.stop()
        self._portable_job = None
        self._closing.emit()
        self._deletion_jobs.clear()
        self._pending_reads.clear()
        self._generation += 1
        self._debounce.stop()
        self.close_detail()
