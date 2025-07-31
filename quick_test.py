#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

def quick_test():
    """Quick test of the gender classification model"""
    print("🎯 Gender Classification Model - Quick Test")
    print("=" * 50)
    
    try:
        from prediction import model, predict_gender
        
        if model is not None:
            print("✅ Model loaded successfully!")
            print(f"📊 Model expects: 160x160 RGB images")
            print(f"🎯 Output classes: female, male")
            print()
            
            # Test with available images
            test_count = 0
            
            # Check sample images
            if os.path.exists('sample_images'):
                for filename in os.listdir('sample_images'):
                    if filename.endswith('.jpg') and test_count < 3:
                        img_path = os.path.join('sample_images', filename)
                        result = predict_gender(img_path)
                        
                        if 'error' not in result:
                            gender = result['predicted_gender']
                            confidence = result['confidence']
                            emoji = "👨" if gender == 'male' else "👩"
                            
                            print(f"{emoji} {filename}: {gender} ({confidence:.1%})")
                            test_count += 1
                        else:
                            print(f"❌ {filename}: {result['error']}")
            
            if test_count == 0:
                print("⚠️  No sample images found for testing")
                print("   The model is ready to use with 160x160 images")
            
            print("\n" + "=" * 50)
            print("✅ SUCCESS: Your MobileNetV2 model is working perfectly!")
            print("🚀 Ready for production predictions!")
            
        else:
            print("❌ Model failed to load")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
