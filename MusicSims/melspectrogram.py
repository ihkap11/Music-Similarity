import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Library To Convert MP3, WAV files into MEL-SPECTROGRAMS")

    parser.add_argument("-i", "--input", required=True, help="Path to input file")
    parser.add_argument("-o", "--output", required=True, help="Path to output PNG")
    parser.add_argument("-ll", "--lowerLimit", help="Lower Limit for Sample (in seconds)", default=0)
    parser.add_argument("-ul", "--upperLimit", help="Upper Limit for Sample (in seconds)")
    parser.add_argument("-n_mels", "--nMELS", help="Number of MEL bins", default=128)
    parser.add_argument("-dB", "--dB", help="Power to dB", default=False)
    parser.add_argument("-fmin", "--minFrequency", help="lowest frequency (in Hz)", default=0.0)
    parser.add_argument("-fmax", "--maxFrequency", help="highest frequency (in Hz)", default=11025.0)
    parser.add_argument("-p", "--plot", help="Plot the MelSpectrogram", default=True)

    return parser.parse_args()

def mel_spectrograms(inputFile, args):
    y, sr = librosa.load(inputFile)
    length = len(y)/sr
    print "Length of Audio File: {}".format(length)
    
    if args.upperLimit is None:
        args.upperLimit = length
    print "Range -- [{}(s) - {}(s)]".format(args.lowerLimit, args.upperLimit)
    y_sample = y[max(0,int(args.lowerLimit)) * sr : min(int(args.upperLimit) * sr, len(y))]
    S = librosa.feature.melspectrogram(y_sample, sr=sr, n_mels=128)
    if args.dB:
        log_S = librosa.power_to_db(S, ref=np.max)
    else:
        log_S = S

    plt.figure(figsize=(30,5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.savefig(args.output)
    if args.plot:
        plt.show()

if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.isfile(args.input):
        raise Exception("Input audio file not found!")
    
    if args.output is None:
        raise Exception("Output Path cannot be Empty")

    mel_spectrograms(args.input, args)
