import requests
import os
import json
from pathlib import Path
import random

def extended_accuracy_test():
    """Test model accuracy with larger sample"""
    base_url = "http://127.0.0.1:8000"
    
    # Get more test samples
    test_cases = []
    
    # Get 25 male test images
    male_dir = Path("data/test/male")
    if male_dir.exists():
        male_files = list(male_dir.glob("*.jpg"))
        random.shuffle(male_files)
        for file in male_files[:25]:
            test_cases.append((str(file), "male"))
    
    # Get 25 female test images  
    female_dir = Path("data/test/female")
    if female_dir.exists():
        female_files = list(female_dir.glob("*.jpg"))
        random.shuffle(female_files)
        for file in female_files[:25]:
            test_cases.append((str(file), "female"))
    
    print(f"🧪 Extended Accuracy Test with {len(test_cases)} images")
    print("=" * 60)
    
    correct = 0
    total = 0
    results = {
        "male": {"correct": 0, "total": 0, "confidences": []}, 
        "female": {"correct": 0, "total": 0, "confidences": []}
    }
    
    for i, (img_path, actual_gender) in enumerate(test_cases):
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
                results[actual_gender]["confidences"].append(confidence)
                
                # Print progress every 10 images
                if (i + 1) % 10 == 0:
                    current_acc = correct / total * 100
                    print(f"Progress: {i+1}/{len(test_cases)} - Current accuracy: {current_acc:.1f}%")
                
            else:
                print(f"❌ Failed: {os.path.basename(img_path)}")
                
        except Exception as e:
            print(f"❌ Error: {os.path.basename(img_path)} - {e}")
    
    # Calculate detailed results
    overall_accuracy = correct / total if total > 0 else 0
    male_accuracy = results["male"]["correct"] / results["male"]["total"] if results["male"]["total"] > 0 else 0
    female_accuracy = results["female"]["correct"] / results["female"]["total"] if results["female"]["total"] > 0 else 0
    
    # Calculate average confidences
    male_avg_conf = sum(results["male"]["confidences"]) / len(results["male"]["confidences"]) if results["male"]["confidences"] else 0
    female_avg_conf = sum(results["female"]["confidences"]) / len(results["female"]["confidences"]) if results["female"]["confidences"] else 0
    
    print("\n" + "=" * 60)
    print("📊 EXTENDED ACCURACY RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy:     {overall_accuracy:.2%} ({correct}/{total})")
    print(f"Male Accuracy:        {male_accuracy:.2%} ({results['male']['correct']}/{results['male']['total']})")
    print(f"Female Accuracy:      {female_accuracy:.2%} ({results['female']['correct']}/{results['female']['total']})")
    print(f"Male Avg Confidence:  {male_avg_conf:.1%}")
    print(f"Female Avg Confidence: {female_avg_conf:.1%}")
    
    return overall_accuracy

if __name__ == "__main__":
    extended_accuracy_test()
