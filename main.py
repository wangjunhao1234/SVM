import subprocess

def extract_handcrafted_features():
    print("Running: Extract Handcrafted Features")
    subprocess.run(["python", "extract_handcrafted_features.py"])

def train_evaluate_svm():
    print("Running: Train and Evaluate SVM")
    subprocess.run(["python", "train_evaluate_svm.py"])

def train_evaluate_ffnn():
    print("Running: Train and Evaluate FFNN")
    subprocess.run(["python", "train_evaluate_ffnn.py"])

def plot_results():
    print("Running: Plot Results")
    subprocess.run(["python", "plot_results.py"])

def main():
    extract_handcrafted_features()
    train_evaluate_svm()
    train_evaluate_ffnn()
    plot_results()

if __name__ == "__main__":
    main()
