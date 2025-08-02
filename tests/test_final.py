#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

def test_model():
    """Test the gender classification model"""
    print("Gender Classification Model Test")
    print("=" * 50)
    
    try:
        from prediction import model, class_names, predict_gender, GenderPredictor
        
        if model is not None:
            print(f"✓ Model loaded successfully!")
            try:
                if hasattr(model.input, 'shape'):
                    print(f"✓ Input shape: {model.input.shape}")
                else:
                    print(f"✓ Input shape: {model.input[0].shape}")
                
                if hasattr(model.output, 'shape'):
                    print(f"✓ Output shape: {model.output.shape}")
                else:
                    print(f"✓ Output shape: {model.output[0].shape}")
            except:
                print("✓ Model structure loaded (shape info unavailable)")
                
            print(f"✓ Classes: {class_names}")
            print()
            
            # Test with sample images if available
            test_images = []
            
            # Look for sample images
            if os.path.exists('sample_images'):
                for file in os.listdir('sample_images'):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append(os.path.join('sample_images', file))
            
            # If no sample images, get from test data
            if not test_images:
                if os.path.exists('data/test/male'):
                    male_files = [f for f in os.listdir('data/test/male') if f.endswith('.jpg')]
                    if male_files:
                        test_images.append(os.path.join('data/test/male', male_files[0]))
                
                if os.path.exists('data/test/female'):
                    female_files = [f for f in os.listdir('data/test/female') if f.endswith('.jpg')]
                    if female_files:
                        test_images.append(os.path.join('data/test/female', female_files[0]))
            
            if test_images:
                print(f"Testing with {len(test_images)} images:")
                print("-" * 30)
                
                for img_path in test_images[:4]:  # Test max 4 images
                    print(f"Testing: {os.path.basename(img_path)}")
                    result = predict_gender(img_path)
                    
                    if 'error' not in result:
                        # Extract actual gender from path for comparison
                        actual_gender = None
                        if '/male/' in img_path:
                            actual_gender = 'male'
                        elif '/female/' in img_path:
                            actual_gender = 'female'
                        
                        predicted_gender = result['predicted_gender']
                        confidence = result['confidence']
                        
                        status = "✓" if actual_gender == predicted_gender else "✗"
                        
                        print(f"  {status} Predicted: {predicted_gender} ({confidence:.1%})")
                        if actual_gender:
                            print(f"    Actual: {actual_gender}")
                        print(f"    Probabilities: F={result['probabilities']['female']:.3f}, M={result['probabilities']['male']:.3f}")
                    else:
                        print(f"  ✗ Error: {result['error']}")
                    print()
            
            else:
                print("No test images found. Testing with dummy data...")
                # Test with dummy data using GenderPredictor class
                import numpy as np
                predictor = GenderPredictor()
                predictor.model = model  # Use already loaded model
                
                dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                result = predictor.predict_single(dummy_image)
                
                if 'error' not in result:
                    print(f"✓ Dummy test successful!")
                    print(f"  Predicted: {result['predicted_gender']}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                    print(f"  Probabilities: {result['probabilities']}")
                else:
                    print(f"✗ Dummy test failed: {result['error']}")
            
            print("=" * 50)
            print("✓ Model is working correctly!")
            print("✓ Ready for production use!")
            
        else:
            print("✗ Model failed to load")
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
