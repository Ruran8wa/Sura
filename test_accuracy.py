import requests
import os
import json
from pathlib import Path

def test_accuracy():
    """Test model accuracy on test dataset"""
    base_url = "http://127.0.0.1:8000"
    
    # Test samples from each class
    test_cases = []
    
    # Get male test images
    male_dir = Path("data/test/male")
    if male_dir.exists():
        male_files = list(male_dir.glob("*.jpg"))[:10]  # Test 10 samples
        for file in male_files:
            test_cases.append((str(file), "male"))
    
    # Get female test images  
    female_dir = Path("data/test/female")
    if female_dir.exists():
        female_files = list(female_dir.glob("*.jpg"))[:10]  # Test 10 samples
        for file in female_files:
            test_cases.append((str(file), "female"))
    
    print(f"Testing {len(test_cases)} images...")
    
    correct = 0
    total = 0
    results = {"male": {"correct": 0, "total": 0}, "female": {"correct": 0, "total": 0}}
    
    for img_path, actual_gender in test_cases:
        try:
            with open(img_path, 'rb') as f:
                files = {'file': (os.path.basename(img_path), f, 'image/jpeg')}
                response = requests.post(f"{base_url}/predict-file", files=files)
            
            if response.status_code == 200:
                result = response.json()
                predicted_gender = result['predicted_gender']
                confidence = result['confidence']
                
                is_correct = predicted_gender == actual_gender
                if is_correct:
                    correct += 1
                    results[actual_gender]["correct"] += 1
                
                total += 1
                results[actual_gender]["total"] += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {os.path.basename(img_path)}: {actual_gender} → {predicted_gender} ({confidence:.1%})")
            else:
                print(f"❌ Failed to predict {os.path.basename(img_path)}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error with {os.path.basename(img_path)}: {e}")
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0
    male_accuracy = results["male"]["correct"] / results["male"]["total"] if results["male"]["total"] > 0 else 0
    female_accuracy = results["female"]["correct"] / results["female"]["total"] if results["female"]["total"] > 0 else 0
    
    print("\n" + "=" * 50)
    print("�� ACCURACY RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    print(f"Male Accuracy:    {male_accuracy:.2%} ({results['male']['correct']}/{results['male']['total']})")
    print(f"Female Accuracy:  {female_accuracy:.2%} ({results['female']['correct']}/{results['female']['total']})")
    
    return overall_accuracy

if __name__ == "__main__":
    test_accuracy()
