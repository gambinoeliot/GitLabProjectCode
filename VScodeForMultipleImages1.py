import pandas as pd 
import matplotlib.pylab as plt
import numpy as np 
import scipy.stats as stats
import os
from glob import glob 
import cv2
import hashlib
from hashlib import blake2b
from scipy.stats import truncnorm
from


def image_file_details(): 
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

#Now generate the random number sequences using the randomness algorithm. We will get one sequence which has been hashed and one sequence which hasn't to compare the effectiveness of hashing the data.

def extract_bits_from_image(img, crop_box=None):
  # # Extract the least significant bit from each pixel.
  # # (For an 8-bit value, doing a bitwise AND with 1 gives the LSB.)
  lsb_array = img & 1
  # # Flatten the 2D array to a 1D bit stream.
  bit_stream = lsb_array.flatten()
  return bit_stream


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

# descriptive statistics from binary array
def des_stats(binary_array):
  N = len(binary_array)
  #find the number of 1's 
  k = sum(binary_array)
  p_mle= k/N
  meanK = p_mle
  varK = N*p_mle*(1-p_mle)
  return N, k, p_mle, meanK, varK

# generate a z-test statistic
def zValue(p_mle, N):
  z = (p_mle - 0.5) / np.sqrt(p_mle*(1-p_mle)/N)
  return z

# compare to standard normal:
def z_accept_reject(z_values):
  # fill in with mathematical formula
  alpha = 0.05 
  lower_bound = -1.96
  upper_bound = 1.96
  conditions = [alpha, lower_bound, upper_bound]
  for i in z_values:
    if i > lower_bound and i < upper_bound:
      conditions.append('Fail to reject null hypothesis')
    else:
      conditions.append('Reject null hypothesis')
  return conditions

# comparison of moments and descriptions against one another and the normal dist ########
# ECDF z-value 
"""
The following z-value is used for the ecdf, to reshape the byte data from the Von Neumann extractor to be within a standard normal configuration.

ALL MENTIONS OF "DATA" REFER TO VON NEUMANN EXTRACTED BITS DATA WHICH HAS BEEN CONVERTED INTO 8-BIT NUMBER VALUES
"""
def z_value_ecdf(byte_data):
  μ = 255/2
  σ = np.sqrt(255) / (1/4)
  z = [(x-μ)/σ for x in byte_data]
  return z 

# plot the ECDF
def plot_ecdf(data, name):
    fig, ax = plt.subplots()

    # Sort the data
    x = np.sort(data)

    # Compute ECDF values
    y = np.arange(1, len(x) + 1) / len(x)

    # Generate the theoretical CDF for a standard normal
    x_cdf = np.linspace(-3, 3, 1000)
    y_cdf = stats.norm.cdf(x_cdf)

    # Plot the ECDF
    ax.step(x, y, where="post", label="Empirical CDF", linewidth=2)

    # Plot the true CDF
    ax.plot(x_cdf, y_cdf, label="Standard Normal CDF", linestyle="dashed", color="red")

    # Labels and legend
    ax.set_xlabel("z-value (x-μ)/σ")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"ECDF of Von Neumann Bits vs Standard Normal CDF for image: {name}")
    ax.legend()
    ax.grid()

    return fig  # Return figure instead of showing it

# Using the KS test
def KStest(data):
  # Find the bounds of the data
  lower, upper = min(data), max(data)
  # Define the cropped normal CDF function
  def cropped_normal_cdf(x):
      norm_cdf = stats.norm.cdf(x)
      lower_cdf = stats.norm.cdf(lower)
      upper_cdf = stats.norm.cdf(upper)
      return (norm_cdf - lower_cdf) / (upper_cdf - lower_cdf)
    
  # Perform the KS test with the cropped normal CDF
  ks_statistic, ks_p_value = stats.kstest(data, cropped_normal_cdf)
  return ks_statistic, ks_p_value

# Anderson-Darling test for normality
def anderson_darling(data):
  anderson_result = stats.anderson(data, dist='norm')
  print(f"Anderson-Darling Statistic: {anderson_result.statistic}")
  return anderson_result.statistic
  #print(f"Critical Values: {anderson_result.critical_values}")

# plot the Von Neumann number values which have been z-shifted as a histogram against a normal distribution
def plotting_hist(data, name):
    fig, ax = plt.subplots()

    ax.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Data')

    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.sqrt(np.var(data, ddof=1)))
    ax.plot(x, p, 'k', linewidth=2, label="Normal fit")

    ax.set_title(f"Histogram and Normal Distribution Fit for image: {name}")
    ax.set_xlabel('z-value')
    ax.set_ylabel('Probability density')
    ax.legend()

    return fig  # Return figure instead of showing it

# ----- Main Execution -----  THIS IS TO BE RUN LAST
if __name__ == '__main__':
  # Specify the directory where your iPhone LED images are stored.
  img_directory = '/Users/eliotgambino/Library/CloudStorage/OneDrive-UniversityofBirmingham/Uni-Documents/Phys-Year2-docs/year2-modules/labProject/PythonCode/LEDimages'
    
  # Define a crop box if needed to isolate the LED.
  # Example: (left, upper, right, lower). Adjust these values to your image.
  # crop_box = (100, 100, 300, 300)  # <-- Change as appropriate

  # Step 0: import the image files and check file directory is accessible
  img_files = glob(img_directory+'/*.JPG')
  print(f'**** the file path has been identified as: {os.path.exists(img_directory)}')
  #print(img_files)

  # define the array which will store all the data we need to save, for analysing the fit of our data against the null hypothesis.
  Data = []
  img_Names = []

  for i in range(len(img_files)):

    file_path = img_files[i]
    img = plt.imread(file_path)
    file_name = os.path.basename(file_path)
    name = file_name.split('.')[0]
    img_Names.append(name)
    print(f'Now analysing the {name} image.')
  
    ## Step 1: import the image file and display its formatting
    # image_details = image_file_details()
    #print(image_details)

    # Step 2: gather the first array of binary values directly from the image, the Von Neumann extracted bits, and the hashed bits
    print('Processing random bits...')
    arrayVals_img = extract_bits_from_image(img)
    arrayVals_VN = von_neumann_extractor(bit_array=arrayVals_img)
    von_byte_data = bits_to_bytes(extracted_bits=arrayVals_VN)[0]
    arrayVals_hash= hash_random_data(von_byte_data=von_byte_data)
    Binary_data = [arrayVals_img, arrayVals_VN, arrayVals_hash]
    print('Random bits successfully generated.')

    # save name of each for file saving
    ArrayVals_img = 'ArrayVals_img'
    ArrayVals_VN = 'ArrayVals_VN'
    ArrayVals_Hash = 'ArrayVals_Hash'
    Names = [ArrayVals_img, ArrayVals_VN, ArrayVals_Hash]

    # Step 3: save the binary values to files in a given directory
    Directory = '/Users/eliotgambino/Library/CloudStorage/OneDrive-UniversityofBirmingham/Uni-Documents/Phys-Year2-docs/year2-modules/labProject/PythonCode/VScodePython/VScodeImageBinaryFiles/'
    for j in range(len(Names)):
      specific_directory = Directory+'binaryValues/'
      save_to_file = specific_directory + Names[j] + '_' + name + ".csv"
      df = pd.DataFrame(Binary_data[j], dtype=int)
      df.to_csv(save_to_file, index=False)
    print(f'Successfully saved {name} binary files.')


    # Step 3: create a z-test statistic for each array and compare to standard normal Hypothesis test.
    print("Generating z-value for image, Von Neumann and Hashed data...")
    stats_imgVals = des_stats(arrayVals_img)
    stats_VNVals = des_stats(arrayVals_VN)
    stats_hashVals = des_stats(arrayVals_hash)

    z_img = zValue(stats_imgVals[2], stats_imgVals[0])
    z_VN = zValue(stats_VNVals[2], stats_VNVals[0])
    z_hash = zValue(stats_hashVals[2], stats_hashVals[0])
    z_values = [z_img, z_VN, z_hash] # this is to be stored as one of the saved data statistics
    conditions = z_accept_reject(z_values)

    print(f"Z-values generated. Image z-value: {z_img}, Von Neumann z-value: {z_VN}, hashed z-value: {z_hash}.")
    print(f'The acceptance region for a value of alpha = {conditions[0]} has lower and upper bounds of: {conditions[1], conditions[2]}. For the Image bits: {conditions[3]}, Von Neumann bits: {conditions[4]}, Hashed bits: {conditions[5]}. This is based on the z-value statistic.')

    fig, axs = plt.subplots(1,2, figsize=(15,5))
    # Step 4: generate 8-bit values (data) for raw, von neumann and hashed bits
    byte_data_VN = bits_to_bytes(extracted_bits=arrayVals_VN)[1]
    byte_data_raw = bits_to_bytes(extracted_bits=arrayVals_img)[1]
    byte_data_hashed = bits_to_bytes(extracted_bits=arrayVals_hash)[1]
    data_VN = z_value_ecdf(byte_data_VN)
    data_Raw = z_value_ecdf(byte_data_raw)
    data_Hashed = z_value_ecdf(byte_data_hashed)

    # Step 5: Create an ECDF plot of the Von Neumann data and plot against a standard normal CDF
    ECDF_plot = plot_ecdf(data_VN, name)
    print(ECDF_plot)

    # Step 5: Implement KS test and Anderson-Darling tests for raw, von neumann and hashed bits
    ks_statistic_VN, ks_p_value_VN = KStest(data_VN)[0], KStest(data_VN)[1]
    ks_statistic_raw, ks_p_value_raw = KStest(data_Raw)[0], KStest(data_Raw)[1]
    ks_statistic_hashed, ks_p_value_hashed = KStest(data_Hashed)[0], KStest(data_Hashed)[1]

    #print(f"KS Statistic: {ks_statistic_VN:.10f}")
    #print(f"KS P-value: {ks_p_value_VN:.10e}")

    AD_results_raw = anderson_darling(data_Raw)
    AD_results_VN = anderson_darling(data_VN)
    AD_results_hashed = anderson_darling(data_Hashed)

    KS = [ks_statistic_raw, ks_statistic_VN, ks_statistic_hashed]
    KSp = [ks_p_value_raw, ks_p_value_VN, ks_p_value_hashed]
    AD_results = [AD_results_raw, AD_results_VN, AD_results_hashed]

    # saving the data to a file for pandas implementation later
    for k in range(3):
      Data.append([z_values[k], KS[k], KSp[k], AD_results[k]])

    #print(AD_results)

    # Step 6: plot a histogram of the z-shifted values from the Von Neumann byte number data against a standard normal
    hist_plot = plotting_hist(data_VN, name)
    print(hist_plot)

    # save the plots
    plots = [ECDF_plot, hist_plot]
    for i, fig in enumerate(plots):
      specific_directory = Directory+'Plots/'
      fig.savefig(f"{specific_directory}plot_{name}_{i+1}.png", dpi=300)
      plt.close(fig)  # Close to free memory

    print(f'Successfully saved {name} plots.')
    print('-' * 50) # for clarity between images

  # Saving the data we have gathered as a CSV file
  cols = ['z-value', 'AD-Statistic', 'KS p-value', 'Anderson Darling Statistic']
  dataFrames = []
  for j in range(0, len(Data), 3): 
    if j+3 == len(Data):
      dataFrames.append(pd.DataFrame(Data[j:], dtype=float, index = [f'Raw Image: {img_Names[j // 3]}', f'Von Neumann: {img_Names[j // 3]}', f'Hashed: {img_Names[j // 3]}'], columns=cols))
    else:
      dataFrames.append(pd.DataFrame(Data[j:j+3], dtype=float, index = [f'Raw Image: {img_Names[j // 3]}', f'Von Neumann: {img_Names[j // 3]}', f'Hashed: {img_Names[j // 3]}'], columns=cols))

  final_df = pd.concat(dataFrames)
  statistic_directory = '/Users/eliotgambino/Library/CloudStorage/OneDrive-UniversityofBirmingham/Uni-Documents/Phys-Year2-docs/year2-modules/labProject/PythonCode/VScodePython/VScodeImageBinaryFiles/VScodeStatisticsOfFit/'
  final_df.to_csv(statistic_directory + 'statisticalValues.csv')
  


#### ADDITIONS ####
# may also be good to compute the ECDF for the hashed and bare binary data
