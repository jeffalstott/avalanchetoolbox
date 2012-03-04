def write_to_HDF5(data, file_name, condition, sampling_rate, \
        window='blackmanharris', taps=513, filter_type='FIR',\
        amplitude=False, displacement_aucs=False, amplitude_aucs=False,\
        overwrite=False,\
        bands = ('raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad'),
        downsample='nyquist',\
        **attrs):
    import h5py

    if downsample==False:
        downsample=sampling_rate
    
    f = h5py.File(file_name+'.hdf5')

    try:
        f[condition]
    except KeyError:
        f.create_group(condition)
        pass

    if 'raw' not in list(f[condition]):
        f.create_group(condition+'/raw')

    HDF5_filter(f[condition], sampling_rate,\
            window=window, taps=taps, filter_type=filter_type,\
            amplitude=amplitude, displacement_aucs=displacement_aucs, amplitude_aucs=amplitude_aucs,\
            overwrite=overwrite,\
            bands = bands,\
            downsample=downsample)

    if 'raw' not in bands:
        del f[condition+'/raw']

    for i in attrs.keys():
        f.attrs[i]=attrs[i]

    f.close()
    return

def HDF5_filter(file, sampling_rate,\
        window='hamming', taps=25, filter_type='FIR',\
        amplitude=False, displacement_aucs=False, amplitude_aucs=False,\
        overwrite=False,\
        bands = ('raw', 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad'),
        downsample='nyquist'):

    from avalanches import area_under_the_curve, fast_amplitude
    from time import gmtime, strftime, clock
    import h5py

    if type(file)!=h5py._hl.group.Group:
        return

    for i in file.keys():
        if i.startswith('filter'):
            continue
        elif not i.startswith('raw'):
            HDF5_filter(file[i])
        else:
            if 'displacement' not in file[i].keys():
                return
            else:
#At this point we know there is a 'raw' directory with a 'displacement' in it,
# so we can filter!
                if downsample==False:
                    downsample=sampling_rate

                version = 'filter_'+filter_type+'_'+str(taps)+'_'+window+'_ds-'+str(downsample)
                if version not in file.keys():
                    file.create_group(version)
                file[version].attrs['filter_type'] = filter_type
                file[version].attrs['window'] = window
                file[version].attrs['taps'] = taps

                data = file['raw/displacement'][:,:]

                for band in bands:
                    print 'Processing '+band
                    if band=='raw':
                        if amplitude and 'amplitude' not in file['raw'].keys():
                            data_amplitude = fast_amplitude(data)
                            file.create_dataset('/raw/amplitude', data=data_amplitude)
                        if displacement_aucs and 'displacement_aucs' not in file['raw'].keys():
                            data_displacement_aucs = area_under_the_curve(data)
                            file.create_dataset('/raw/displacement_aucs', data=data_displacement_aucs)
                        if amplitude_aucs and 'amplitude_aucs' not in file['raw'].keys():
                            data_amplitude_aucs = area_under_the_curve(data_amplitude)
                            file.create_dataset('/raw/amplitude_aucs', data=data_amplitude_aucs)
                        continue

                    if band not in file[version].keys():
                        file.create_group(version+'/'+band)

                    if 'displacement' not in file[version+'/'+band].keys():
                        print 'Filtering, '+str(data.shape[-1])+' time points' 
                        filtered_data, frequency_range, downsampled_rate = band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window, downsample=downsample)
                        file.create_dataset(version+'/'+band+'/displacement', data=filtered_data)
                    elif overwrite:
                        print 'Filtering, '+str(data.shape[-1])+' time points' 
                        filtered_data, frequency_range, downsampled_rate = band_filter(data, band, sampling_rate=sampling_rate, taps=taps, window_type=window, downsample=downsample)
                        file.create_dataset(version+'/'+band+'/displacement', data=filtered_data)
                    elif amplitude_aucs or amplitude or displacement_aucs:
                        filtered_data = file[version+'/'+band+'/displacement'][:,:]
                    else:
                        continue

                    if amplitude and 'amplitude' not in file[version+'/'+band].keys():
                        print 'Fast amplitude, '+str(filtered_data.shape[-1])+' time points'
                        tic = clock()
                        data_amplitude = fast_amplitude(filtered_data)
                        file.create_dataset(version+'/'+band+'/amplitude', data=data_amplitude)
                        toc = clock()
                        print toc-tic
                    elif amplitude_aucs:
                        data_amplitude = file[version+'/'+band+'/amplitude'][:,:]

                    if displacement_aucs and 'displacement_aucs' not in file[version+'/'+band].keys():
                        print 'Area under the curve, displacement'
                        tic = clock()
                        data_displacement_aucs = area_under_the_curve(filtered_data)
                        file.create_dataset(version+'/'+band+'/displacement_aucs', data=data_displacement_aucs)
                        toc = clock()
                        print toc-tic

                    if amplitude_aucs and 'amplitude_aucs' not in file[version+'/'+band].keys():
                        print 'Area under the curve, amplitude'
                        tic = clock()
                        data_amplitude_aucs = area_under_the_curve(data_amplitude)
                        file.create_dataset(version+'/'+band+'/amplitude_aucs', data=data_amplitude_aucs)
                        toc = clock()
                        print toc-tic

                    file[version+'/'+band].attrs['frequency_range'] = frequency_range
                    file[version+'/'+band].attrs['downsampled_rate'] = downsampled_rate
                    file[version+'/'+band].attrs['processing_date'] = strftime("%Y-%m-%d", gmtime())
                return

def band_filter(data, band, sampling_rate=1000.0, taps=25.0, window_type='hamming', downsample=False):
    """docstring for band_filter"""
    from numpy import array, floor
    #Some neuroscience specific frequency bands
    bands = {'delta': (array([1.0, 4.0]), False),
            'theta': (array([4.0,8.0]), False),
            'alpha': (array([8.0,12.0]), False),
            'beta': (array([12.0,30.0]), False),
            'gamma': (array([30.0,80.0]), False),
            'high-gamma': (array([80.0, 100.0]), False),
            'broad': (array([1.0, 100.0]), False),
            }
    if band=='raw':
        return data
    if type(band)==str:
        frequencies = bands[band][0]
        pass_zero = bands[band][1]
    else:
        frequencies = band[0]
        pass_zero = band[1]
    from scipy.signal import firwin, lfilter
    nyquist = sampling_rate/2.0
    kernel= firwin(taps, frequencies/nyquist, pass_zero=pass_zero, window = window_type)
    data = lfilter(kernel, 1.0, data)
    if downsample==True or downsample=='nyquist':
        downsampling_interval = floor(( 1.0/ (2.0*frequencies.max()) )*sampling_rate)
    elif downsample:
        downsampling_interval = floor(sampling_rate/float(downsample))
    else:
        return data, frequencies, sampling_rate

    data = data[:,::downsampling_interval]
    downsampled_rate=sampling_rate/downsampling_interval
    return data, frequencies, downsampled_rate

def phaseshuffle(input_signal):
    """phaseshuffle(input_signal) phaseshuffle shuffles the phases of the component frequencies of a real signal among each other, but preserving the phase of the DC component and any Nyquist element. Input is a matrix of channels x timesteps."""

    ## Fourier Transform to Get Component Frequencies' Phases and Magnitudes
    length_of_signal = input_signal.shape[1]
    from numpy.fft import rfft, irfft
    from numpy import angle, zeros, concatenate, exp
    from numpy.random import permutation
    print("Calculating component frequencies and their phases")
    y = rfft(input_signal, axis=1)
    magnitudes = abs(y)
    phases = angle(y)

## Shuffle Phases, Preserving DC component and Nyquist element (if present)
    number_of_channels, N = y.shape
    randomized_phases = zeros(y.shape)

    print("Randomizing")
    for j in range(number_of_channels):

        if N & 1: #If there are an odd number of elements
            #Retain the DC component and shuffle the remaining components.
            order = concatenate(([0], permutation(N-1)+1), axis=1)
        else:
            #Retain the DC and Nyquist element component and shuffle the remaining components. This makes the new signal real, instead of complex.
            order = concatenate(([0], permutation(N-2)+1, [-1]), axis=1)

        randomized_phases[j] = phases[j, order]

## Construct New Signal
    print("Constructing new signal")
    y1 = magnitudes*exp(1j*randomized_phases)
    output_signal = irfft(y1,n=length_of_signal, axis=1) #While the above code will produce a real signal when given a real signal, numerical issues sometimes make the output from ifft "complex", with a +/- 0i component. Since the imaginary component is accidental and meaningless, we remove it.
    return output_signal
