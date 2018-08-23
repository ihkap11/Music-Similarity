"""
Script converts mp3 audio files to 5 sec 128x224 spectrograms
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import os
from tqdm import tqdm
import pylab

def parse_arguments():
    parser = argparse.ArgumentParser(description="Library To Convert MP3, WAV files into MEL-SPECTROGRAMS")

    parser.add_argument("-i", "--input", help="/upload", default ="upload" )
    parser.add_argument("-o", "--output",  help="Path to output folder", default = "spectrograms")
    parser.add_argument("-off", "--offset", help="Point to begin from (in seconds)", default=0)
    parser.add_argument("-dur", "--duration", help="Duration of song from the offset (in seconds)", default = 5)
    parser.add_argument("-crop", "--cropSeconds", help="cropping off seconds from beginning and ending of song", default=15)  
    parser.add_argument("-n_mels", "--nMELS", help="Number of MEL bins", default=128)
    parser.add_argument("-fmin", "--minFrequency", help="lowest frequency (in Hz)", default=0.0)
    parser.add_argument("-fmax", "--maxFrequency", help="highest frequency (in Hz)", default=11025.0)    
#     parser.add_argument("-pr", "--progress", help="Prints progress rate", default=True)
#     parser.add_argument("-p", "--plot", help="Plot the MelSpectrogram", default=True)

    return parser.parse_args()


def mel_spectrograms(folderpath, off, duration, n_mels, fmin, fmax, out, crop ):
   
#     print("sss",folderpath)
    
    for root, dirnames, filenames in os.walk(folderpath):
            print("dd",filenames)
            i = 0
            for filename in tqdm(filenames): 
                
                filepath=os.path.join(root, filename ).replace('\\', '/')
                print(filepath)
                classs = filename.split('.')[0]
                audio_name = filename.split('.')[0]
#                 print("an",classs)
                i = i+1
#                 print(out)
                spec_out=os.path.join(out+"/"+ classs+"/"+audio_name).replace('\\', '/')
               
#                 print("******",spec_out)
                extract_features(filepath,  off, duration, n_mels, fmin, fmax, spec_out, crop )
    

def extract_features(audio_loc, off, duration, n_mels, fmin, fmax, spec_out, crop ): 
    offset = 0.0
    duration = 5
    count=0
    
    total_y, sr = librosa.load(audio_loc)
    total_y = total_y[15 * sr : -(15 *sr)]
    print(spec_out, count)
    while True:
        y = total_y[int(offset * sr) : int((offset+duration) * sr)]
        if len(y) <= 0:
            break

        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, n_fft=550, hop_length=220, fmin=0, fmax=11025.0)
        log_S = librosa.power_to_db(S, ref=np.max)
        if log_S.shape[1] < 224:
            break
            
        save_plot(log_S, spec_out, count)
        
        offset += duration
        count += 1   
            
        
        
def save_plot(feature, output_loc, part):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2,os
    
    # saves to 128 x 224 dimensional image   
    pylab.figure(figsize=(2.24,1.28))
#     pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
    
    librosa.display.specshow(feature,cmap='gray_r')
#     print("))))))))))",output_loc +".png")
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)   
        
#     print("*************** INSIDE", output_loc +"_"+str(part)+".png")
    pylab.savefig(output_loc +"_"+str(part)+".png", transparent = True, bbox_inches = None, pad_inches = 0, dpi = 100)
    pylab.close()
    
    

                
def main():
    args = parse_arguments()
#     print("RUM")
    mel_spectrograms(args.input,  args.offset, args.duration, args.nMELS, args.minFrequency, args.maxFrequency, args.output, args.cropSeconds)    
    
    
    
if __name__ == "__main__":
    main()