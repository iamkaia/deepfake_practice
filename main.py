# main.py
from train import train_lora, train_baseline
from evaluate import evaluate_and_log
from explain import visualize_errors

def main():
    train_lora()
    train_baseline()
    evaluate_and_log()
    visualize_errors()

if __name__ == "__main__":
    main()
