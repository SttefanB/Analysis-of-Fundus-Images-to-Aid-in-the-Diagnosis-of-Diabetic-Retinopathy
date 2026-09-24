import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (512, 512)

def inferenta_cascada(gate_model, severity_model, img_paths, gate_threshold=0.5, batch_size=32):
    n = len(img_paths)
    gate_proba = np.zeros(n)
    clase_finale = np.zeros(n, dtype=int)
    
    # PAS 1: Gate Model — Clasificare Binară
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = []
        for path in img_paths[start:end]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = preprocess_input(img.astype(np.float32))
            batch.append(img)
            
        preds = gate_model.predict(np.array(batch), verbose=0).flatten()
        gate_proba[start:end] = preds
        
    gate_bolnav = gate_proba >= gate_threshold
    clase_finale[~gate_bolnav] = 0 
    idx_bolnavi = np.where(gate_bolnav)[0]
    
    # PAS 2: Severity Model — Clasificare Multiclasă
    for start in range(0, len(idx_bolnavi), batch_size):
        end = min(start + batch_size, len(idx_bolnavi))
        batch_idx = idx_bolnavi[start:end]
        batch = []
        
        for i in batch_idx:
            img = cv2.imread(img_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = preprocess_input(img.astype(np.float32))
            batch.append(img)
            
        preds = severity_model.predict(np.array(batch), verbose=0)
        
        for j, idx in enumerate(batch_idx):
            clase_finale[idx] = np.argmax(preds[j])
            
    return clase_finale, gate_proba