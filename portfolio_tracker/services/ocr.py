"""OCR local y parser tolerante para comprobantes de GBM+."""

from __future__ import annotations

import io
import importlib.util
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from pytesseract import Output

from ..config import LOCAL_TIMEZONE


MONTHS = {
    "ene": 1,
    "feb": 2,
    "mar": 3,
    "abr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "ago": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dic": 12,
}


class OcrUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class OcrExtraction:
    raw_text: str
    confidence: Decimal | None
    fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_ocr_text(text: str) -> str:
    """Repara ruido OCR frecuente en etiquetas, sin inferir valores numéricos.

    Algunos motores devuelven el carácter de reemplazo en vocales acentuadas
    (``Operaci�n``) o pierden la ``i`` de ``Títulos``. Solo se normalizan las
    etiquetas conocidas; títulos, importes y símbolos siguen viniendo del
    comprobante y conservan la revisión humana obligatoria.
    """

    normalized = _ascii(text).replace("\ufffd", "")
    repairs = (
        (r"\boperaci(?:o|0)?n\b", "operacion"),
        (r"\bcomisi(?:o|0)?n\b", "comision"),
        (r"\bt[i1l]?tulos\b", "titulos"),
        (r"\bt[i1l]?tulo\b", "titulo"),
    )
    for pattern, replacement in repairs:
        normalized = re.sub(
            pattern, replacement, normalized, flags=re.IGNORECASE
        )
    return normalized


def _decimal(text: str | None) -> Decimal | None:
    if not text:
        return None
    cleaned = re.sub(r"[^0-9,.-]", "", text)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        tail = cleaned.rsplit(",", 1)[-1]
        cleaned = cleaned.replace(",", ".") if len(tail) <= 2 else cleaned.replace(",", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _match(text: str, pattern: str, group: int = 1) -> str | None:
    found = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return found.group(group).strip() if found else None


def parse_gbm_text(text: str) -> OcrExtraction:
    """Convierte texto OCR variable en campos editables, nunca en una orden ciega."""

    compact = re.sub(r"\s+", " ", _normalize_ocr_text(text)).strip()
    fields: dict[str, Any] = {}
    warnings: list[str] = []

    fields["symbol"] = _match(
        compact,
        r"(?:emisora|ticker)\s*[:\-]?\s*([A-Z][A-Z0-9.\-]{0,14})(?=\s|$)",
    )
    fields["product"] = _match(
        compact,
        r"producto\s*[:\-]?\s*(USA|SIC|MEXICO|GLOBAL)(?=\s|$)",
    )
    side = _match(
        compact,
        r"tipo\s+de\s+operacion\s*[:\-]?\s*(compra|venta)(?=\s|$)",
    )
    fields["side"] = side.capitalize() if side else None
    order_type = _match(
        compact,
        r"tipo\s+de\s+orden\s*[:\-]?\s*(limitada|mercado|stop(?:\s+limit)?)(?=\s|$)",
    )
    fields["order_type"] = order_type.capitalize() if order_type else None
    fields["quantity"] = _decimal(
        _match(compact, r"titulos\s*[:\-]?\s*([0-9][0-9.,]*)")
    )
    fields["price_usd"] = _decimal(
        _match(
            compact,
            r"precio\s+por\s+titulo\s*[:\-]?\s*\$?\s*([0-9][0-9.,]*)",
        )
    )

    completed_window = _match(
        compact,
        r"orden\s+completada(.{0,100}?)(?=\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|sept|oct|nov|dic)|producto|$)",
    )
    fields["reported_total_usd"] = _decimal(
        _match(completed_window or "", r"\$?\s*([0-9][0-9.,]*)\s*USD")
    )
    # En el formato GBM adjunto, el encabezado corresponde al total bruto y
    # la comision aparece separada. El usuario puede cambiar esta interpretacion.
    fields["reported_total_type"] = "GROSS"

    commission = re.search(
        r"comision\s*[:\-]?\s*([0-9][0-9.,]*)\s*%[^0-9]{0,12}\$?\s*([0-9][0-9.,]*)\s*USD",
        compact,
        flags=re.IGNORECASE,
    )
    if commission:
        fields["commission_rate_pct"] = _decimal(commission.group(1))
        fields["commission_usd"] = _decimal(commission.group(2))
    else:
        fields["commission_rate_pct"] = _decimal(
            _match(compact, r"comision\s*[:\-]?\s*([0-9][0-9.,]*)\s*%")
        )
        fields["commission_usd"] = _decimal(
            _match(compact, r"comision.{0,45}?\$\s*([0-9][0-9.,]*)\s*USD")
        )

    dates = list(
        re.finditer(
            r"(\d{1,2})\s+(ene|feb|mar|abr|may|jun|jul|ago|sep|sept|oct|nov|dic)\s+(\d{4})"
            r"(?:\s*[-–—]\s*(\d{1,2})[:.]([0-9]{2}))?",
            _ascii(text),
            flags=re.IGNORECASE,
        )
    )
    if dates:
        selected = next((item for item in dates if item.group(4)), dates[0])
        fields["executed_at"] = datetime(
            int(selected.group(3)),
            MONTHS[selected.group(2).lower()],
            int(selected.group(1)),
            int(selected.group(4) or 0),
            int(selected.group(5) or 0),
            tzinfo=LOCAL_TIMEZONE,
        )
    else:
        fields["executed_at"] = None

    required = ("symbol", "side", "quantity", "price_usd")
    missing = [field_name for field_name in required if not fields.get(field_name)]
    if missing:
        labels = {
            "symbol": "emisora",
            "side": "tipo de operación",
            "quantity": "títulos",
            "price_usd": "precio por título",
        }
        warnings.append(
            "Revisa manualmente: faltan "
            + ", ".join(labels[field_name] for field_name in missing)
            + "."
        )
    return OcrExtraction(raw_text=text, confidence=None, fields=fields, warnings=warnings)


class TesseractGbmExtractor:
    """Motor ligero. El ejecutable Tesseract se instala una sola vez en Windows."""

    KEYWORDS = ("orden", "emisora", "operacion", "titulos", "precio", "comision")

    def __init__(self) -> None:
        configured = os.getenv("TESSERACT_CMD")
        candidates = [
            configured,
            shutil.which("tesseract"),
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        command = next((item for item in candidates if item and Path(item).exists()), None)
        if command:
            pytesseract.pytesseract.tesseract_cmd = str(command)

    def is_available(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except (pytesseract.TesseractNotFoundError, OSError):
            return False

    @staticmethod
    def _prepare(content: bytes) -> Image.Image:
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.width < 1200:
            scale = 1200 / image.width
            image = image.resize(
                (1200, int(image.height * scale)), Image.Resampling.LANCZOS
            )
        gray = ImageOps.grayscale(image)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = ImageEnhance.Contrast(gray).enhance(1.35)
        return gray.filter(ImageFilter.SHARPEN)

    def _run(self, image: Image.Image, psm: int) -> tuple[str, Decimal | None]:
        language = "spa+eng"
        try:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                config=f"--oem 3 --psm {psm}",
                output_type=Output.DICT,
            )
        except pytesseract.TesseractError:
            # Algunas instalaciones solo incluyen ingles; el parser no depende
            # de diccionario, por lo que sigue funcionando razonablemente.
            data = pytesseract.image_to_data(
                image,
                lang="eng",
                config=f"--oem 3 --psm {psm}",
                output_type=Output.DICT,
            )
        words: list[str] = []
        confidences: list[Decimal] = []
        last_line: tuple[int, int, int] | None = None
        lines: list[str] = []
        current: list[str] = []
        for index, token in enumerate(data["text"]):
            token = token.strip()
            if not token:
                continue
            line_key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
            if last_line is not None and line_key != last_line and current:
                lines.append(" ".join(current))
                current = []
            current.append(token)
            words.append(token)
            last_line = line_key
            try:
                confidence = Decimal(str(data["conf"][index]))
                if confidence >= 0:
                    confidences.append(confidence)
            except InvalidOperation:
                pass
        if current:
            lines.append(" ".join(current))
        average = (
            sum(confidences, Decimal("0")) / len(confidences)
            if confidences
            else None
        )
        return "\n".join(lines), average

    def extract(self, content: bytes) -> OcrExtraction:
        if not self.is_available():
            raise OcrUnavailableError(
                "Tesseract OCR no esta instalado. Puedes capturar la operacion "
                "manualmente o instalarlo y volver a analizar."
            )
        image = self._prepare(content)
        candidates = [self._run(image, 6), self._run(image, 11)]

        def score(candidate: tuple[str, Decimal | None]) -> tuple[int, Decimal]:
            normalized = _ascii(candidate[0]).lower()
            keyword_score = sum(word in normalized for word in self.KEYWORDS)
            return keyword_score, candidate[1] or Decimal("0")

        text, confidence = max(candidates, key=score)
        result = parse_gbm_text(text)
        result.confidence = confidence.quantize(Decimal("0.1")) if confidence else None
        if confidence is not None and confidence < 65:
            result.warnings.append("La confianza OCR es baja; confirma todos los importes.")
        return result


class RapidOcrGbmExtractor:
    """Respaldo autocontenido ONNX; se carga solamente al analizar una imagen."""

    def is_available(self) -> bool:
        return (
            importlib.util.find_spec("rapidocr") is not None
            and importlib.util.find_spec("onnxruntime") is not None
        )

    def extract(self, content: bytes) -> OcrExtraction:
        if not self.is_available():
            raise OcrUnavailableError("RapidOCR no esta instalado.")
        import numpy as np
        from rapidocr import RapidOCR

        image = TesseractGbmExtractor._prepare(content).convert("RGB")
        # Limitar hilos y desactivar el arena de CPU reduce el pico de recursos
        # en una aplicacion personal que analiza una captura a la vez.
        engine = RapidOCR(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
            }
        )
        output = engine(np.asarray(image))
        texts = tuple(getattr(output, "txts", ()) or ())
        scores = tuple(getattr(output, "scores", ()) or ())
        if not texts:
            raise OcrUnavailableError("El OCR no encontro texto legible en la imagen.")
        text = "\n".join(str(item) for item in texts)
        confidence = None
        if scores:
            confidence = (
                sum((Decimal(str(item)) for item in scores), Decimal("0"))
                / len(scores)
                * Decimal("100")
            )
        result = parse_gbm_text(text)
        result.confidence = confidence.quantize(Decimal("0.1")) if confidence else None
        if confidence is not None and confidence < 65:
            result.warnings.append("La confianza OCR es baja; confirma todos los importes.")
        return result


class GbmOcrExtractor:
    """Selecciona Tesseract si existe y RapidOCR autocontenido como respaldo."""

    def __init__(self) -> None:
        self.tesseract = TesseractGbmExtractor()
        self.rapid = RapidOcrGbmExtractor()

    def is_available(self) -> bool:
        return self.tesseract.is_available() or self.rapid.is_available()

    @property
    def backend_name(self) -> str:
        if self.tesseract.is_available():
            return "Tesseract local"
        if self.rapid.is_available():
            return "RapidOCR local (ONNX)"
        return "No disponible"

    def extract(self, content: bytes) -> OcrExtraction:
        if self.tesseract.is_available():
            return self.tesseract.extract(content)
        return self.rapid.extract(content)
