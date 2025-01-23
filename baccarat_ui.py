import pandas as pd
from joblib import load

def predict_next_outcome():
    """
    Get user input for past outcomes and predict next result
    """
    # Load saved model and mapping
    model = load('baccarat_model.joblib')
    outcome_mapping = load('outcome_mapping.joblib')
    
    print("\nPredict Next Baccarat Outcome")
    print("Enter past 5 outcomes (P=Player, B=Banker, T=Tie)")
    
    past_outcomes = []
    for i in range(5):
        while True:
            try:
                outcome = input(f"Outcome {i+1}: ").upper()
                if outcome in ['P', 'B', 'T']:
                    past_outcomes.append(outcome)
                    break
                print("Invalid input. Use P, B, or T")
            except ValueError:
                print("Invalid input. Use P, B, or T")

    # Create features
    features = pd.DataFrame({
        f'Prev_{i+1}': [outcome_mapping[outcome]] 
        for i, outcome in enumerate(past_outcomes)
    })
    features['Streak_Length'] = 1

    # Get prediction and probabilities
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    # Map prediction back to label
    pred_label = {v: k for k, v in outcome_mapping.items()}[pred]
    
    print(f"\nPredicted Next Outcome: {pred_label}")
    print(f"Probabilities: Player={probs[0]:.2f}, Banker={probs[1]:.2f}, Tie={probs[2]:.2f}")

if __name__ == "__main__":
    predict_next_outcome()