import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null); // Состояние для превью
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const fileInputRef = useRef(null);

  const handlePredict = async () => {
    if (!text || !image) return alert("Please enter text and select an image!");
    
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append('text', text);
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setResult(response.data);
    } catch (err) {
      alert("Error connecting to the server. Please check if app.py is running.");
    } finally {
      setLoading(false);
    }
  };

  const onPickFile = () => {
    fileInputRef.current.click();
  };

  // Функция обработки выбора файла с генерацией превью
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file)); // Создаем временный URL для тега img
    }
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <h1 className="main-title">
          Multimodálna detekcia príspevkov sociálnych sietí s obsahom vygenerovaných GenAI
        </h1>
        <p className="subtitle">Multimodal text and image analysis (CLIP)</p>
        
        <div className="upload-card">
          <textarea 
            className="text-input"
            placeholder="Enter a description or post text..." 
            onChange={(e) => setText(e.target.value)}
          />
          
          <div className="file-input-container">
            <label className="file-input-label">Upload image:</label>
            
            <input 
              type="file" 
              accept="image/*" 
              ref={fileInputRef}
              style={{ display: 'none' }} 
              onChange={handleFileChange} 
            />

            {/* Блок превью изображения */}
            {preview && (
              <div className="image-preview-wrapper">
                <img src={preview} alt="Selected preview" className="image-preview" />
              </div>
            )}

            <button className="custom-file-btn" onClick={onPickFile}>
              {image ? "Change Image" : "Choose File"}
            </button>

            <div className="file-name-text">
              {image ? image.name : "No file chosen"}
            </div>
          </div>

          <button className="submit-btn" onClick={handlePredict} disabled={loading}>
            {loading ? "Analyzing..." : "Check for AI"}
          </button>

          {result && (
            <div className={`result-box ${result.is_ai ? 'result-ai' : 'result-human'}`}>
              <h2 style={{ margin: 0, fontSize: '1.4rem' }}>Analysis result</h2>
              <div className="progress-container">
                <div 
                  className="progress-bar" 
                  style={{ 
                    width: result.ai_probability, 
                    backgroundColor: parseFloat(result.ai_probability) > 50 ? '#ffa726' : '#66bb6a' 
                  }}
                ></div>
              </div>
              <div className="probability-text" style={{ color: result.is_ai ? '#e65100' : '#2e7d32' }}>
                Probability of AI: {result.ai_probability}
              </div>
              <p style={{ fontSize: '20px', fontWeight: '600' }}>
                {result.is_ai ? "🤖 AI Generated" : "👤 Human Made"}
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="footer-info">
        <div>Created by <b>Denys Lutsenko</b></div>
        <div>Consultant <b>Ing. Viliam Balara</b></div>
        <div>Head <b>Prof. Ing. Kristína Machová, PhD.</b></div>
      </div>
    </div>
  );
}

export default App;