# Re-export public API from the templates module.
#
# The ``templates/`` package directory exists to hold static assets
# (HTML templates for risk-heatmap, threat-hunt playbook, etc.).
# The actual template logic lives in ``_templates_impl.py`` (formerly
# ``templates.py`` before the package was created).  This shim keeps
# ``from replication.templates import ContractTemplate`` working.

from .._templates_impl import (  # noqa: F401
    ContractTemplate,
    TEMPLATES,
    get_template,
    list_templates,
    get_categories,
    render_catalog,
    render_comparison_table,
    main,
)
