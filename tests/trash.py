# Fix fitz import to resolve type checking issues
try:
    from fitz import Document as PyMuPDFDocument, Page as PyMuPDFPage

    # Type aliases for type checking
    Document = PyMuPDFDocument  # type: ignore[assignment]
    Page = PyMuPDFPage  # type: ignore[assignment]
except ImportError:
    # For type hinting only
    class Document:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> "Page":
            raise NotImplementedError

        @property
        def metadata(self) -> dict:
            return {}

    class Page:
        def get_text(self, text_type: str) -> str:
            return ""
