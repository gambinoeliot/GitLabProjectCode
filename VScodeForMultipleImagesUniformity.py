import pandas as pd 
import matplotlib.pylab as plt
import numpy as np 
import scipy.stats as stats
import os
from glob import glob 
import cv2
import hashlib
from hashlib import blake2b



def image_file_details(img): 
  #briefly check the image file format. Using just the green channel, hence [:,:,1]
  print(f'***** The image array:{img[:,:,1]}')
  print(f'**** The image format MPL: {img.shape}.')
  print(f'**** The maximum colour intensity value is: {np.max(img[:,:,1])}')

  #flatten the array into a 1D array of pixel values. Deduct any over a threshold intensity of 230
  img_array = pd.Series(img[:,:,1].flatten())
  flat_array = [x for x in img_array if x <= 230]

  #plotting the actual image, the cropped array data and the actual array data
  figSetup, axsetup = plt.subplots(1,3, figsize=(15,5))
  axsetup[0].imshow(img)
  axsetup[0].axis('off')
  axsetup[0].set_title('Picture of LED')
  pd.Series(flat_array).plot(kind='hist', bins=100, title='Distribution of Pixel Values.', ax=axsetup[1])
  pd.Series(img[:,:,1].flatten()).plot(kind='hist', bins=100, title='Distribution of Pixel Values.', ax=axsetup[2])
  plt.show()

# Now generate the random number sequences using the randomness algorithm. We will get one sequence which has been hashed and one sequence which hasn't to compare the effectiveness of hashing the data.

def extract_bits_from_image(img):
  # # Extract the least significant bit from each pixel.
  # # (For an 8-bit value, doing a bitwise AND with 1 gives the LSB.)
  lsb_array = img & 1
  # # Flatten the 2D array to a 1D bit stream.
  bit_stream = lsb_array.flatten()
  return bit_stream

# Now shuffle the binary values to scramble the order. This removes any underlying pattern in the phone camera.
def shuffle_bits(bit_array):
  return np.random.permutation(bit_array)

"""
Implements the von Neumann extractor:
- Processes the bit array in pairs.
- If a pair is 01, output 0; if 10, output 1; else, discard.
"""
def von_neumann_extractor(bit_array):
  extracted_bits = []
  # Process in steps of 2 bits
  for i in range(0, len(bit_array) - 1, 2):
    first, second = bit_array[i], bit_array[i+1]
    if first == 0 and second == 1:
        extracted_bits.append(0)
    elif first == 1 and second == 0:
        extracted_bits.append(1)
    # Discard if pair is 00 or 11
  return np.array(extracted_bits)


# now take the von neumann bits and convert them to bytes before hashing them
"""
Optionally hashes the random data to produce a uniform output.
Converts an array of bits (0s and 1s) into a bytes object.
    
The bit array length is adjusted to a multiple of 8.

Two lists are returned. One is a byted version, one is a list of numbers from 0 to 255. The latter is used in the ECDF.
"""
def bits_to_bytes(extracted_bits):
  # Ensure the total number of bits is a multiple of 8.
  n = len(extracted_bits) - (len(extracted_bits) % 8)
  bit_array_8 = extracted_bits[:n]
  # Convert the bit array to a string of bits.
  bit_str = ''.join(str(bit) for bit in bit_array_8)
  # Break the string into chunks of 8 and convert each to an integer.
  byte_list = [int(bit_str[i:i+8], 2) for i in range(0, len(bit_str), 8)]
  return bytes(byte_list), byte_list

def hash_random_data(von_byte_data):
  byted = blake2b(von_byte_data).hexdigest()
  res = ''.join(format(ord(i), '08b') for i in byted)
  res = list(map(int, res))
  #printing result 
  return res

############# now the probability distributions for both 0 and 1 for each bit sequence we have.

def cumulative_plot_uniform(data, name, Type): ### CHANGE WIDTH OF FIG
  fig, ax = plt.subplots()
  E = lambda k: (np.sum(np.array(data) < k)) / len(data)
  EC = np.zeros(256)

  for x in range(0,256):
    EC[x] = E(x)

  ax.step(range(0,256),EC, color='b', label= f'{Type} Byte Data')
  ax.step(range(0,256),np.linspace(0,1,256), color='red', label='Randomly Generated Uniform Data.')

  ax.set_title(f"Cumulative Distribution for Image: {name} against True Uniform.")
  ax.set_xlabel(f'8-bit {Type} Value')
  ax.set_ylabel('Cumulative Probability density')
  ax.legend()

  return fig, EC  # Return figure instead of showing it. Also return the EC for use in max deviation.

# define a maximum deviation for the data CDF and the expected Uniform CDF
def max_deviation(EC):
  max_dev = np.max(np.abs(EC - np.linspace(0,1,256))) #maximum deviation 
  return max_dev

def plotting_hist(data, name, Type): ### CHANGE WIDTH OF FIG
    fig, ax = plt.subplots()

    ax.hist(data, bins=100, density=True, alpha=0.6, color='b', label=f'{Type} Byte Data PDF')

    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin-5, xmax+5, 100)
    ax.plot(x, stats.uniform.pdf(x, loc=xmin, scale=256), color="red", label='Expected Continuous Uniform PDF')
  
    ax.set_title(f"{Type} Byte Data Histogram and Uniform Distribution Fit for image: {name} with 100 bins.")
    ax.set_xlabel(f'8-bit {Type} Value')
    ax.set_ylabel('Probability density')
    ax.legend()

    return fig  # Return figure instead of showing it


# ----- Main Execution -----  THIS IS TO BE RUN LAST
if __name__ == '__main__':
  # Specify the directory where your iPhone LED images are stored.
  #img_directory = '/Users/eliotgambino/Library/CloudStorage/OneDrive-UniversityofBirmingham/Uni-Documents/Phys-Year2-docs/year2-modules/labProject/PythonCode/LEDimages'
  img_directory = '/Users/eliotgambino/Library/CloudStorage/OneDrive-UniversityofBirmingham/Uni-Documents/Phys-Year2-docs/year2-modules/labProject/PythonCode/VScodePython/VScodeExperimentalimages'

  # Step 0: import the image files and check file directory is accessible
  img_files = glob(img_directory+'/*.JPG')
  print(f'**** the file path has been identified as: {os.path.exists(img_directory)}')

  # define the array which will store all the data we need to save, for analysing the fit of our data against the null hypothesis.
  Data = []
  img_Names = []

  for i in range(len(img_files)):
  #for i in range(5): # this is just a test to see if all goes well

    file_path = img_files[i]
    img = plt.imread(file_path)
    file_name = os.path.basename(file_path)
    name = file_name.split('.')[0]
    img_Names.append(name)
    print(f'Now analysing the {name} image.')
  
    ## Step 1: import the image file and display its formatting
    # image_details = image_file_details()
    #print(image_details)

    # Step 2: gather the first array of binary values directly from the image, then shuffled them, then Von Neumann extracted bits, and then hashed bits
    print('Processing random bits...')
    arrayVals_img = extract_bits_from_image(img)
    shuffled_bits = shuffle_bits(arrayVals_img)
    arrayVals_VN = von_neumann_extractor(bit_array=shuffled_bits)
    von_byte_data = bits_to_bytes(extracted_bits=arrayVals_VN)[0]
    arrayVals_hash= hash_random_data(von_byte_data=von_byte_data)
    Binary_data = [arrayVals_img, arrayVals_VN, arrayVals_hash]
    print('Random bits successfully generated.')

    # useful for later in saving data
    types = ['Raw Image', 'Von Neumann', 'Hashed']

    # save name of each for file saving
    ArrayVals_img = 'ArrayVals_img'
    ArrayVals_VN = 'ArrayVals_VN'
    ArrayVals_Hash = 'ArrayVals_Hash'
    Names = [ArrayVals_img, ArrayVals_VN, ArrayVals_Hash]

    # Step 3: save the binary values to files in a given directory
    save_directory = '/Users/eliotgambino/Library/CloudStorage/OneDrive-UniversityofBirmingham/Uni-Documents/Phys-Year2-docs/year2-modules/labProject/PythonCode/VScodePython/saved_data_3/'

    # Step 4: generate 8-bit values (data) for raw, von neumann and hashed bits
    byte_data_VN = bits_to_bytes(extracted_bits=arrayVals_VN)[1]
    byte_data_raw = bits_to_bytes(extracted_bits=arrayVals_img)[1]
    byte_data_hashed = bits_to_bytes(extracted_bits=arrayVals_hash)[1]

    # step 5: produce a cumulative plot and a pdf plot for the VN bits against a uniform dist.
    cumulative_uniform_plot_Raw = cumulative_plot_uniform(byte_data_raw, name, Type = 'Raw')[0]
    pdf_uniform_plot_Raw = plotting_hist(byte_data_raw, name, Type = 'Raw')
    cumulative_uniform_plot_VN = cumulative_plot_uniform(byte_data_VN, name, Type = 'VN')[0]
    pdf_uniform_plot_VN = plotting_hist(byte_data_VN, name, Type = 'VN')
    cumulative_uniform_plot_Hash = cumulative_plot_uniform(byte_data_hashed, name, Type = 'Hashed')[0]
    pdf_uniform_plot_Hash = plotting_hist(byte_data_hashed, name, Type = 'Hashed')


    # Step 6: Calculate the maximum deviation of our data from the Uniform CDF
    EC_raw = cumulative_plot_uniform(byte_data_raw, name, Type='Raw')[1]
    EC_VN = cumulative_plot_uniform(byte_data_VN, name, Type = 'VN')[1]
    EC_hashed = cumulative_plot_uniform(byte_data_hashed, name, Type = 'Hashed')[1]

    max_dev_raw = max_deviation(EC_raw)
    max_dev_VN = max_deviation(EC_VN)
    max_dev_hashed = max_deviation(EC_hashed)
    max_devs = [max_dev_raw, max_dev_VN, max_dev_hashed]
    Data.append(max_devs)

    # Step 7: save the plots
    plots = [cumulative_uniform_plot_VN, pdf_uniform_plot_VN]
    for i, fig in enumerate(plots):
      specific_directory = save_directory+'Plots/'
      fig.savefig(f"{specific_directory}plot_{name}_{i+1}.png", dpi=300)
      plt.close(fig)  # Close to free memory

    print(f'Successfully saved {name} plots.')
    print('-' * 50) # for clarity between images

  # Saving the data we have gathered as a CSV file
  cols = ['Maximum Deviation']
  dataFrames = []
  for j in range(0, len(img_Names)): # for each group of three max_deviations for raw,VN,hash...
    # want to add the respective maximum deviation to raw, VN and hashed
    for t in range(3):
      dataFrames.append(pd.DataFrame(Data[j][t], dtype=float, index = [f'{types[t]}: {img_Names[j]}'], columns=cols))

  final_df = pd.concat(dataFrames)
  statistic_directory = save_directory + 'statistics/'
  final_df.to_csv(statistic_directory + 'statisticalValues.csv')
