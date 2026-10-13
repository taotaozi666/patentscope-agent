"""文档加载入口：通过扩展名注册加载器，统一返回 LangChain Document。"""

from collections.abc import Callable, Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from app.pdf.loader import NativePdfLoader


class UTF8DocumentLoader(BaseLoader):
    """读取上传的 TXT/Markdown 字节，无需先保存临时文件。"""

    def __init__(self, source: str, content: bytes) -> None:
        self.source = source
        self.content = content

    def lazy_load(self) -> Iterator[Document]:
        yield Document(
            page_content=self.content.decode("utf-8-sig"),
            metadata={
                "source": self.source,
                "format": Path(self.source).suffix.lower(),
            },
        )


LoaderFactory = Callable[[str, bytes], BaseLoader]


class LoaderRegistry:
    """后续 PDF、Word 等加载器只需注册，无需修改导入业务。"""

    def __init__(self) -> None:
        self._factories: dict[str, LoaderFactory] = {}

    def register(self, extension: str, factory: LoaderFactory) -> None:
        self._factories["." + extension.lower().lstrip(".")] = factory

    @property
    def extensions(self) -> list[str]:
        return [extension.lstrip(".") for extension in sorted(self._factories)]

    def load(self, source: str, content: bytes) -> list[Document]:
        extension = Path(source).suffix.lower()
        if extension not in self._factories:
            raise ValueError(f"暂不支持 {extension or '无扩展名'} 文件")
        return self._factories[extension](source, content).load()


loaders = LoaderRegistry()
loaders.register("txt", UTF8DocumentLoader)
loaders.register("md", UTF8DocumentLoader)
loaders.register("pdf", NativePdfLoader)
