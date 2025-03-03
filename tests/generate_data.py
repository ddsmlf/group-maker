import pandas as pd
import random
from datetime import datetime, timedelta
import names

def generate_random_name():
    return f"{names.get_last_name()}-{names.get_first_name()}"

def generate_test_csv(file_path, num_students=100, max_wishes=5):
    students = [generate_random_name() for _ in range(num_students)]
    students = list(set(students))  # Remove duplicates
    num_students = len(students)
    
    data = []
    start_time = datetime(2025, 2, 28, 18, 0, 0)
    
    for student in students:
        timestamp = start_time + timedelta(seconds=random.randint(1, 300))
        wishes = random.sample([s for s in students if s != student], random.randint(1, max_wishes))
        wishes_str = " ".join(wishes)
        data.append([timestamp.strftime("%m/%d/%Y %H:%M:%S"), wishes_str, student])
    
    df = pd.DataFrame(data, columns=["Timestamp", "Vœux", "nom-prenom"])
    df.to_csv(file_path, index=False)
    print(f"Fichier de test généré: {file_path}")

if __name__ == "__main__":
    generate_test_csv("test_reponses.csv")