"""First-class multimodal argument types for function tools.

``File`` and ``Image`` are framework-owned types that developers can
declare in function-tool signatures. The LLM sees them as a string
parameter (URL or path); the framework reconstructs the typed object
before invoking the developer's function.

Use case::

    from troopai.adk.types.multimodal import Image
    from troopai.adk.tools import function_tool


    @function_tool(name="ocr", description="Extract text from an image.")
    def ocr(image: Image) -> str:
        if image.url is not None:
            data = _fetch(image.url)
        elif image.path is not None:
            data = image.path.read_bytes()
        else:
            data = image.data
        return _ocr(data, image.mime_type or "image/png")

The LLM sees the tool schema as ``{"image": {"type": "string",
"format": "uri"}}`` and calls with a URL or path string; the
framework constructs the ``Image`` instance from that string before
``ocr`` runs.

For binary payloads the developer constructs at runtime — e.g.
returning an ``Image`` from a sibling tool that produced a PNG —
:meth:`File.as_data_url` builds an inline ``data:<mime>;base64,…``
URL that downstream serializers can carry through provider message
boundaries.
"""

from troopai.adk.types.multimodal.file import File, Image

__all__ = ["File", "Image"]
