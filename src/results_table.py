def print_results_table(results):
    print("\n" + "=" * 85)
    print(f"{'Model':<35} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("=" * 85)

    for result in results:
        print(
            f"{result['model']:<35} "
            f"{result['rmse']:<12.4f} "
            f"{result['mae']:<12.4f} "
            f"{result['mape']:<12.4f}"
        )

    print("=" * 85)