import numpy as np
import pandas as pd
import random

def generate_synthetic_data(num_users=1000, num_sessions=5000):
    # Define possible values for each feature
    user_ids = [f"user_{i}" for i in range(1, num_users + 1)]
    ages = np.random.randint(18, 65, size=num_users)  # Random ages between 18 and 65
    genders = np.random.choice(["Male", "Female", "Other"], size=num_users, p=[0.48, 0.48, 0.04])
    locations = np.random.choice(["US", "Europe", "Asia", "Other"], size=num_users, p=[0.5, 0.2, 0.2, 0.1])

    # Create a DataFrame for user info
    user_data = pd.DataFrame({
        "user_id": user_ids,
        "age": ages,
        "gender": genders,
        "location": locations
    })

    # Define session data
    session_ids = [f"session_{i}" for i in range(1, num_sessions + 1)]
    session_user_ids = np.random.choice(user_ids, size=num_sessions)
    session_durations = np.random.normal(loc=5, scale=2, size=num_sessions).clip(0.5, 15)  # Session times in minutes
    devices = np.random.choice(["Desktop", "Mobile", "Tablet"], size=num_sessions, p=[0.5, 0.4, 0.1])

    # Define events
    event_types = ["click", "view", "search", "purchase"]
    event_probs = [0.4, 0.4, 0.15, 0.05]  # Probability of each event type
    events = np.random.choice(event_types, size=num_sessions, p=event_probs)

    # Create a DataFrame for session data
    session_data = pd.DataFrame({
        "session_id": session_ids,
        "user_id": session_user_ids,
        "session_duration": session_durations,
        "device": devices,
        "event_type": events
    })

    # Simulate ads data
    ad_categories = ["Electronics", "Clothing", "Home", "Beauty", "Toys"]
    ad_data = {
        "ad_category": [],
        "ad_cost": [],
        "conversion_rate": [],
        "click_rate": []
    }
    for category in ad_categories:
        ad_data["ad_category"].append(category)
        ad_data["ad_cost"].append(round(random.uniform(0.5, 5.0), 2))  # Random CPC
        ad_data["conversion_rate"].append(round(random.uniform(0.01, 0.15), 2))  # Conversion rate
        ad_data["click_rate"].append(round(random.uniform(0.2, 0.8), 2))  # Click-through rate

    ads_data = pd.DataFrame(ad_data)

    # Merge session and user data
    merged_data = pd.merge(session_data, user_data, on="user_id")
    
    return merged_data, ads_data

if __name__ == "__main__":
    synthetic_data, ads_data = generate_synthetic_data()
    
    # Save data to CSV for later use
    synthetic_data.to_csv("data/processed/synthetic_data.csv", index=False)
    ads_data.to_csv("data/processed/ads_data.csv", index=False)

    print("Synthetic data generated and saved!")

