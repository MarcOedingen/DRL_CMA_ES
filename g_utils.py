from prettytable import PrettyTable


def print_pretty_table(func_dimensions, func_instances, results):
    table = PrettyTable()
    table.field_names = [
        "Function",
        "Dimensions",
        "Instance",
        "Difference (x_best - x_opt)",
    ]
    for i in range(24):
        table.add_row(
            [
                f"Function {i + 1}",
                func_dimensions[i],
                func_instances[i],
                f"{results[i]:.18f}",
            ]
        )
    print(table)
