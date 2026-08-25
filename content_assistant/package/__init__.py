"""Content Packages: one loadable file per book, and the registry over them.

The pipeline's output is per-lesson stage artifacts, which is right for a
pipeline and wrong for a consumer. This package is the boundary between the
two: :func:`load_content` hands an adaptive engine one checked object, and
:class:`ContentRegistry` answers what exists across several of them.

Nothing here generates content. A package is assembled from artifacts by
:mod:`content_assistant.package.build` and can always be rebuilt from them.
"""

from content_assistant.package.migrate import (  # noqa: F401
    SchemaVersionError,
    check_version,
)
from content_assistant.package.registry import ContentRegistry  # noqa: F401
from content_assistant.package.schema import (  # noqa: F401
    BUILDER_VERSION,
    PACKAGE_FILENAME,
    ContentPackage,
    ContentPackageError,
    PackageStats,
    compute_stats,
    default_path,
    load_content,
    save_content,
)
