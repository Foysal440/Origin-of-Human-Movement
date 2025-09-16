from PyQt6.QtWidgets import QTableWidgetItem

def fill_table(table_widget, data):
    """
    Fill a QTableWidget with a 2D list of data.
    """
    table_widget.setRowCount(len(data))
    table_widget.setColumnCount(len(data[0]) if data else 0)
    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            table_widget.setItem(row_idx, col_idx, item)

def update_metrics_display(metrics_widgets, optical_flow_data, hopkins_history):
    """
    Update metric widgets with optical flow and Hopkins statistic data.
    metrics_widgets: dict of {metric_name: widget}
    optical_flow_data: dict of {metric_name: value}
    hopkins_history: list of float
    """
    for key, widget in metrics_widgets.items():
        value = optical_flow_data.get(key, "")
        widget.setText(str(value))
    if "hopkins" in metrics_widgets and hopkins_history:
        metrics_widgets["hopkins"].setText(f"{hopkins_history[-1]:.3f}")

def update_fluid_analysis_table(table_widget, fluid_summary):
    """
    Update a QTableWidget with fluid analysis summary.
    fluid_summary: list of dicts or 2D list
    """
    if isinstance(fluid_summary, list) and fluid_summary and isinstance(fluid_summary[0], dict):
        headers = list(fluid_summary[0].keys())
        data = [[row[h] for h in headers] for row in fluid_summary]
        table_widget.setColumnCount(len(headers))
        table_widget.setHorizontalHeaderLabels(headers)
    else:
        data = fluid_summary
    fill_table(table_widget, data)
