import matplotlib.pyplot as plt


def plot_predictions(
        y_true,
        y_pred,
        title="Prediction vs Actual",
        save_path=None
):
    plt.figure(figsize=(12, 6))

    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Log Return")

    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_predictions_zoom(
        y_true,
        y_pred,
        title="Prediction vs Actual Zoom",
        n_points=50,
        save_path=None
):
    plt.figure(figsize=(12, 6))

    plt.plot(y_true[:n_points], label="Actual")
    plt.plot(y_pred[:n_points], label="Predicted")

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Scaled Log Return")

    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()