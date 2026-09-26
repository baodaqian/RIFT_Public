# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 08:41:28 2020

@author: asligar
"""

'''
Last customized by Daqian for the ChirpIQ data 02/21/24
'''

import numpy as np
import csv
import glob
import os
from collections import OrderedDict

"""
def get_results_files(path,var_name='time_var'):

    # Modification by DB
    # Construct the path in a cross-platform manner
    search_path = os.path.join(path, '**', '*.csv')
    #print(search_path)
    index_file_full_path = glob.glob(search_path, recursive=True)
    
    if len(index_file_full_path) > 1:
        index_file_full_path = os.path.abspath(index_file_full_path[0])
        print(f'WARNING: Multiple index files found, using {index_file_full_path}')
    elif len(index_file_full_path) == 1:
        index_file_full_path = os.path.abspath(index_file_full_path[0])
        print(f'INFO: Index file found, using {index_file_full_path}')
    else:
        print('ERROR: FRTM Index file not found, check path')
        return
    '''
    index_file_full_path = glob.glob(path + '\\**\\*.csv',recursive=True)
    if len(index_file_full_path)>1:
        index_file_full_path = os.path.abspath(index_file_full_path[0])
        print(f'WARNING: Multiple index files found, using {index_file_full_path}')
    elif len(index_file_full_path)==1:
        index_file_full_path = os.path.abspath(index_file_full_path[0])
        print(f'INFO: Index file found, using {index_file_full_path}')
    else:
        print('ERROR: FRTM Index file not found, check path')
        return
        '''
    
    base_path, index_filename = os.path.split(index_file_full_path)
    all_sol_files = glob.glob(base_path + '\\*.frtm')
    
    # DB: the following is specific to the current data
    #print(f'Looking for .frtm files in: {base_path}')  # Debugging: Verify the base path

    # Correctly construct the search path for .frtm files
    all_sol_files = glob.glob(os.path.join(base_path, 'ChirpIQ_KNG4JP_Sweep_Scattered_DV*.frtm'))
    
    if not all_sol_files:
        print("ERROR: No .frtm files found in the specified directory.")
        return None  # No .frtm files found, exit or handle appropriately

    # Assuming you proceed with the first .frtm file found
    path, file_name = os.path.split(all_sol_files[0])
    file_name_prefix = file_name.split("_DV")[0]
    
    '''
    path, file_name = os.path.split(all_sol_files[0])
    file_name_prefix = file_name.split("_DV")[0]
    '''
    
    var_IDS = []
    var_name = 'time_var'
    var_vals = []
    with open(index_file_full_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            if row["Var_ID"] not in var_IDS:
                var_IDS.append(row["Var_ID"])
                if 's' in row[var_name]:
                    val = float(row[var_name].replace("s",""))
                else:
                    val = float(row[var_name])
                var_vals.append(val)
            #print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
            line_count += 1

    
    variation_var_IDS = sorted(zip(var_vals,var_IDS))

    all_frtm =[]  
    all_frtm_dict = {}
    for var_val, id_num in variation_var_IDS:
        #all_frtm[var_val]=f'{path}/{file_name_prefix}_DV{id_num}.frtm'
        all_frtm.append(f'{path}/{file_name_prefix}_DV{id_num}.frtm')
        all_frtm_dict[var_val] = f'{path}/{file_name_prefix}_DV{id_num}.frtm'
        
    print(f'Variations Found: {len(all_frtm)}.')
    all_frtm_dict = OrderedDict(sorted(all_frtm_dict.items()))
    return all_frtm_dict
"""

## Rewritten to address the data for the purpose of INR
def get_results_files(path):
    search_path = os.path.join(path, '**', '*.csv')
    index_file_full_path = glob.glob(search_path, recursive=True)
    
    if len(index_file_full_path) > 1:
        index_file_full_path = os.path.abspath(index_file_full_path[0])
        print(f'WARNING: Multiple index files found, using {index_file_full_path}')
    elif len(index_file_full_path) == 1:
        index_file_full_path = os.path.abspath(index_file_full_path[0])
        print(f'INFO: Index file found, using {index_file_full_path}')
    else:
        print('ERROR: FRTM Index file not found, check path')
        return None
    
    base_path, _ = os.path.split(index_file_full_path)
    all_sol_files = glob.glob(os.path.join(base_path, '*.frtm'))

    var_IDS = []
    var_vals_dphi = []
    var_vals_dtheta = []
    with open(index_file_full_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["Var_ID"] not in var_IDS:
                var_IDS.append(row["Var_ID"])
                # Assuming 'dphi' and 'dtheta' are always present and are floating point numbers
                dphi_val = float(row['dphi']) if 'dphi' in row else 0.0
                dtheta_val = float(row['dtheta']) if 'dtheta' in row else 0.0
                var_vals_dphi.append(dphi_val)
                var_vals_dtheta.append(dtheta_val)

    # Combine dphi and dtheta with Var_ID to ensure unique keys
    variation_var_info = sorted(zip(var_vals_dphi, var_vals_dtheta, var_IDS))

    all_frtm_dict = {}
    for dphi_val, dtheta_val, id_num in variation_var_info:
        key = f'dphi_{dphi_val}_dtheta_{dtheta_val}_ID_{id_num}'
        frtm_file_path = os.path.join(base_path, f'ChirpIQ_KNG4JP_Sweep_Scattered_DV{id_num}.frtm')
        all_frtm_dict[key] = frtm_file_path
        
    print(f'Variations Found: {len(all_frtm_dict)}.')
    return OrderedDict(sorted(all_frtm_dict.items(), key=lambda x: x[0]))

class read_frtm(object):
    """
    reads frtm data output in .transient file
    some of the loops are hardcoded, in the future I will read them
    """
    def __init__(self,filepath):
        """
        reads frtm file from path
        """

        string_to_stop_reading_header = '@ BeginData'
        header = []    
        with open(filepath, "rb") as binary_file:
            line = binary_file.readline()
            line_str = line.decode('ascii')
            while string_to_stop_reading_header not in line_str:
                
                header.append(line_str)
                line = binary_file.readline()
                line_str = line.decode('ascii')
                if line_str.replace(" ","") =="":
                    pass
                elif 'DlxCdVersion' in line_str:
                    dlxcd_vers_line = line_str
                    vers = dlxcd_vers_line.split("=")
                    self.dlxcd_vers = vers
                elif '@ RowCount' in line_str:
                    c = line_str.split("=")
                    c = c[1].replace("\n","").replace("\"","").replace(" ","")
                    self.row_count =  int(c)

                elif '@ ColumnCount' in line_str:
                    c = line_str.split("=")
                    c = c[1].replace("\n","").replace("\"","").replace(" ","")
                    self.col_count =  int(c)
                elif '@ ColHead1' in line_str:
                    c = line_str.split("=")

                    c = c[1].split(" ")
                    c = [i for i in c if i]
                    c = c[0].replace("\n","").replace("\"","").replace(" ","")
                    self.col_header1 =  c
                elif '@ ColHead2' in line_str:
                    c = line_str.split("=")

                    c = c[1].split(" ")
                    c = [i for i in c if i]
                    c = c[0].replace("\n","").replace("\"","").replace(" ","")
                    self.col_header2 =  c
                elif '@ BinaryRecordLength ' in line_str:
                    bin_record_length_line = line_str  
                    c = bin_record_length_line.split("=")
                    self.bin_record_lenth = c[1]
                elif '@ BinaryStartByte ' in line_str:
                    c = line_str.split("=")
                    c = c[1].replace("\n","").replace("\"","").replace(" ","")
                    self.binary_start_byte = int(c)
                elif '@ BinaryRecordSchema ' in line_str:
                    self.bin_byte_type_line = line_str
                elif '@ RadarWaveform ' in line_str:
                    radarwaveform_line = line_str
                    rw = radarwaveform_line.split("=")
                    self.radar_waveform = rw[1].replace("\n","").replace("\"","").replace(" ","" )
                elif '@ RadarChannels ' in line_str:
                    radarchannels_line = line_str
                    rc = radarchannels_line.split("=")
                    self.radarchannels =  rc[1].replace("\n","").replace("\"","").replace(" ","" )
                elif '@ TimeSteps ' in line_str:
                    time_steps_line = line_str
                    c = time_steps_line.split("=")
                    c = c[1].split(" ")
                    c = [i for i in c if i]
                    #debug code to incorporate the INR data formulation
                    if len(c) >= 3:
                        try:
                            self.time_start = float(c[1].replace("\"", ""))  # Now using the second element for time_start
                            self.time_stop = float(c[2].replace("\"", ""))  # And the third for time_stop
                        except ValueError as e:
                            print(f"Error converting time_start or time_stop to float in file: {filepath}")
                            print(f"Content: {c}")
                            print(f"Exception: {e}")
                            # Handle error appropriately, maybe set to None or a default value
                            self.time_start = None
                            self.time_stop = None
                    else:
                        print(f"Unexpected format for time steps in file: {filepath}")
                    #self.time_start= float(c[0].replace("\"",""))
                    #self.time_stop= float(c[1].replace("\"",""))  
                    c = time_steps_line.split("=")
                    c = c[1].split(" ")
                    c = [i for i in c if i]
                    #self.ntime= int(c[2].replace("\"",""))+1
                    try:
                        # First, attempt to convert to float, then to int, if this logic is appropriate
                        self.ntime = int(float(c[2].replace("\"", ""))) + 1
                    except ValueError as e:
                        print(f"Error converting ntime to int in file: {filepath}")
                        print(f"Content: {c}")
                        print(f"Exception: {e}")
                        self.ntime = None 
                    self.time_sweep = np.linspace(self.time_start,self.time_stop,num=self.ntime)
                    if self.ntime is not None and self.ntime > 1:
                        self.time_delta = self.time_sweep[1] - self.time_sweep[0]
                        self.time_duration = self.time_sweep[-1] - self.time_sweep[0]
                    else:
                        # Handle scenarios where self.ntime <= 1
                        self.time_delta = None  # or set to 0 or another default value as appropriate
                        self.time_duration = 0  # Assuming time_duration is the span from start to stop; adjust logic as needed
                    
                    #self.time_delta= self.time_sweep[1]-self.time_sweep[0]
                    #self.time_duration =  self.time_sweep[-1]-self.time_sweep[0]
                elif '@ FreqDomainType ' in line_str:
                    freq_dom_type_line = line_str
                    self.freq_dom_type = freq_dom_type_line.split("=")[1]
                elif '@ FreqSweep ' in line_str:
                    freq_sweep_line = line_str  
                    c = freq_sweep_line.split("=")
                    c = c[1].replace("\"","")
                    c = c.lstrip()
                    c = c.rstrip()
                    c = c.split(" ")
                    c = [i for i in c if i]
                    self.freq_start =  float(c[0])
                    self.freq_stop = float(c[1])
                    self.nfreq = int(c[2].replace("\"",""))+1
                    if (self.radar_waveform == 'CS-FMCW' and self.radarchannels == 'I'):
                        self.nfreq = int(self.nfreq/2)
                    self.freq_sweep = np.linspace(self.freq_start,self.freq_stop,num=self.nfreq)
                    self.freq_delta = self.freq_sweep[1]-self.freq_sweep[0]
                    self.freq_bandwidth = self.freq_sweep[-1]-self.freq_sweep[0]
                    center_index = int(self.nfreq/2)
                    self.freq_center = float(self.freq_sweep[center_index])
                elif '@ AntennaNames ' in line_str:
                    ant_names_line = line_str
                    c = ant_names_line.split("=")
                    c = c[1].replace("\n","")
                    c = c.replace("\"","").replace(" ","" )
                    an = c.split(';')
                    self.antenna_names = an
                elif '@ CouplingCombos '  in line_str:
                    coupling_combos_line = line_str
                    c = coupling_combos_line.replace("\"","")
                    c = c.replace("\n","")
                    c = c.split("=")[1]
                    c = c.split(" ")
                    c = [i for i in c if i]
                    self.num_channels = int(c[0])
                    self.coupling_combos = c[1].split(";")
            self.filepath = filepath
       
        #this is the order in the frtm file
           
        self.channel_names = []
        for each in self.coupling_combos:
            index_values = each.split(',')
            rx_idx =index_values[0]
            tx_idx =index_values[1]
            if ":" in rx_idx:
                rx_idx = int(rx_idx.split(':')[0])
            if ":" in tx_idx:
                tx_idx = int(tx_idx.split(':')[0])
            tx_idx = int(tx_idx)-1
            rx_idx = int(rx_idx)-1
            self.channel_names.append(self.antenna_names[rx_idx] + ":"+ self.antenna_names[tx_idx] )
   
        if self.col_count==2:
            dt = np.dtype([(self.col_header1, float), (self.col_header2, float)])
        else:
            dt = np.dtype([(self.col_header1, float)])   
        raw_data = np.fromfile(filepath, dtype=dt,offset =self.binary_start_byte )
       

        self.all_data = {}
        
        #cdat_real = np.moveaxis(cdat_real,-1,0)
        if self.col_count==2:
            cdat_real = np.reshape(raw_data[self.col_header1],(self.num_channels,self.ntime,self.nfreq))
            cdat_imag = np.reshape(raw_data[self.col_header2],(self.num_channels,self.ntime,self.nfreq))
        #cdat_imag = np.moveaxis(cdat_imag,-1,0)
            for n, ch in enumerate(self.channel_names):
                self.all_data[ch] = cdat_real[n]+cdat_imag[n]*1j
        else:
            cdat_real = np.reshape(raw_data[self.col_header1],(self.num_channels,self.ntime,int(self.nfreq*2))) #fmcw I channel
            for n, ch in enumerate(self.channel_names):
                temp = cdat_real[n]
                #temp2 = temp.T[0::2] #every other sample for fmcw I channel only
                #temp2 = temp.T[0:80]
                self.all_data[ch] = temp
                



    def range_period(self):
        rr = self.range_resolution()
        max_range = rr*self.nfreq
        return max_range
    def range_resolution(self):
        bw = self.freq_bandwidth
        rr = 299792458.0/2/bw
        return rr
    def velocity_resolution(self):
        fc = self.freq_center
        tpt= self.time_duration
        vr = 299792458.0/(2*fc*tpt)
        return vr
    def velocity_period(self):
        vr = self.velocity_resolution()
        np = self.ntime
        vp = np*vr
        return vp

    def load_data(self,order='FreqPulse'):
        '''

        Parameters
        ----------
        order : TYPE, optional
            DESCRIPTION. order is either 'FreqPulse' or 'PulseFreq'. this is
            the index order of the array, [numFreq][numPulse] or [numPulse][numFreq]
            many of the post processing depend on this order so just choose accordingly

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if order.lower()=='freqpulse':
            for ch in self.all_data.keys():
                self.all_data[ch] = self.all_data[ch].T
            return self.all_data
        else: #this is how the data is read in with [pulse][freq] order
            return self.all_data
