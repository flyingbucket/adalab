from src.utils import train_and_save, load_compressed

if __name__ == "__main__":
    (
        clf,
        monitor,
        data_prep,
        (X_train, X_test, y_train, y_test, noise_idx, clean_idx),
        paths,
    ) = train_and_save("./configs/noise5_est500_depth2_v1.json")
