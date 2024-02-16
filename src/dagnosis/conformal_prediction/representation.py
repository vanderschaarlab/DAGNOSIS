# third party
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def representation_class_based(
    train,
    samples,
    n_components=2,
    rep_type="pca",
    seed=42,
):
    """
    Computes a representation of the data using PCA
    """

    scaler = StandardScaler()
    scaler.fit(train)

    combined_X_train_sc = scaler.transform(train)

    samples_sc = scaler.transform(samples)

    if rep_type == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        pcs_train = pca.fit_transform(combined_X_train_sc)
        pcs_samples = pca.transform(samples_sc)

    else:
        raise ValueError("Only PCA is supported")

    return pcs_train, pcs_samples, pca, scaler
