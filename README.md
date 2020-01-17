# Monaural Speech Enhancement using Modified VQ-VAE-2 based on Dictionary Learning
Model architecture is available in `model.py`.

<div align="center">
<img src="figs/archit.png" width="600">
</div>

##### Encoder network architecture
<div align="center">
<img src="figs/model.png" width="600">
</div>

## Speech Enhancement model based on Dictionary Learning
#### Installation
```
pip install -r requirements.txt
```
Load TIMIT corpus and DEMAND dataset, preprocessing for training and validation:
```
bash dataset.sh
```
##### Pre-training
You can train the models as follows:
```
python run.py --pretrain
```
##### Main-training
```
python run.py --maintrain
```
If you want to change the training parameters, please refer to `run.py`.

##### Testing
```
bash checkpoint.sh [input_snr] [epoch]
```

#### Results

<!-- ##### Loss (Pre-training of speech model)
<div align="center">
<img src="loss/speech_model.png" width="700">
</div> -->

##### Modified 1-by-F filters (Pre-training of speech model)
<div align="center">
<img src="figs/filter.gif" width="700">
</div>

<!-- ##### Loss (Main-training of enhancement model)
<div align="center">
<img src="loss/enhancement_model.png" width="700">
</div> -->

##### Outputs qualitative comparison on the log spectrograms of the test speaker
<div align="center">
<img src="figs/result3.png" width="700">
</div>

##### Quantitative comparison of the test speakers
<div align="center">
<img src="figs/table.png" width="700">
</div>

##### Sound samples
##### male speaker BPM0 (SNR 5dB)
- [Clean](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Noisy](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (Proposed)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (FSEGAN)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)

##### male speaker BPM0 (SNR 0dB)
- [Clean](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Noisy](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (Proposed)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (FSEGAN)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)

##### female speaker MLD0 (SNR 5dB)
- [Clean](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Noisy](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (Proposed)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (FSEGAN)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)

##### female speaker MLD0 (SNR 0dB)
- [Clean](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Noisy](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (Proposed)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
- [Enhanced (FSEGAN)](https://drive.google.com/drive/folders/1GllMBAEhLWdKlmWLGSiN0raKI1YXSFlD)
