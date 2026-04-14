const express = require('express');
const { spawn } = require('child_process');
const app = express();

app.use(express.json());

app.post('/predict', (req, res) => {
    const text = req.body.text;
    const python = spawn('python', ['-c', 
        `import joblib; model = joblib.load('../models/model.pkl'); print(model.predict(['${text}'])[0])`
    ]);

    python.stdout.on('data', (data) => {
        const result = data.toString().trim();
        res.json({ 
            text: text, 
            sentiment: result === "1" ? "Positive" : "Negative" 
        });
    });
});

app.listen(3000, () => console.log('API running on http://localhost:3000'));
