"""Order browsing and dashboard support."""

from dal import repo

_DEFAULT_SORT = "created_at"


def search_orders(sort_column=None, direction="DESC"):
    """Returns the order grid sorted the way the dashboard asked."""
    column = sort_column or _DEFAULT_SORT
    return repo.list_orders_sorted(column, direction)


def order_details(order_id):
    """Fetches one order for the detail pane."""
    record = repo.get_order(order_id)
    if record is None:
        return {"error": "not found"}
    return record


def status_summary():
    """Counts per lifecycle stage for the dashboard header."""
    return {
        status: repo.count_by_status(status)
        for status in ("open", "paid", "shipped", "cancelled")
    }
