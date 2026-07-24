from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
    }