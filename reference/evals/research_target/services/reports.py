"""Report rendering for the exports screen."""

import config

_DEFAULT_TEMPLATE = "Report for {report[period]}: {report[total]} orders."


def render_summary(template, report):
    """Fills a user-supplied layout string with report fields."""
    layout = template or _DEFAULT_TEMPLATE
    return layout.format(report=report, settings=config.SETTINGS)


def render_footer():
    """Static footer appended to every export."""
    return config.get("report_footer", "")
