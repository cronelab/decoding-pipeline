# from convert_bci2000 import convert_dat, convert_bcistream

import os
import sys
import argparse
import h5py as h5
import numpy as np
import tqdm

import re

from .h5eeg import H5EEGFile, H5EEGGroup, H5EEGDataset, H5EEGAuxDataset, H5EEGEvents
# TODO Restore pbar to its former glory
# from pbar import ProgressTracker
from .filereader import bcistream

# Use this function to convert a list of datfiles to an hdf5 file
def convert_dat( bcistreams, h5filename = None, relabelcsv = None, overwrite = False, add_everything = False, no_dc = False ):

    # Parse the csv file into a label dictionary
    labeldict = {}
    if relabelcsv is not None:
        csv_f = open( relabelcsv, 'r' )
        for line in csv_f:
            parsed = line.strip().split( ',' )
            labeldict[ parsed[0] ] = parsed[1]

    # Convert the dat file
    datfiles = [bcistreams]
    h5file = convert_bcistream( datfiles, h5filename = h5filename,
        labeldict = labeldict, overwrite = overwrite, add_everything = add_everything, no_dc = no_dc )
    for datfile in datfiles: datfile.close()
    return h5file

# Given a list of bcistreams, use this to convert to an ßhdf5 file.
def convert_bcistream( bcistreams, h5filename = None, labeldict = {}, overwrite = False, add_everything = False, no_dc = False ):

    # If we don't have a filename for the resulting hdf5 file, we will
    # just make an hdf5 file from the current filename in the same directory
    if h5filename == None:
        filepath, h5filename = os.path.split( bcistreams[0].filename )
        h5filename = os.path.splitext( h5filename )[0]
        h5filename = os.path.join( filepath, h5filename + '.hdf5' )

    if os.path.isfile( h5filename ) and not overwrite:
        print( "Error: %s exists already.  Not overwriting." % h5filename )
        return None

    # Create the required group and set group attributes
    # NOTE: BCI2000 file format has no good record of experiment
    outfile = H5EEGFile( h5filename, 'w' )
    group = H5EEGGroup.create( outfile, 
        subject = bcistreams[0].params[ 'SubjectName' ],
        timestamp = bcistreams[0].datestamp )

    # Create the EEG, AUX, and Event datasets
    eeg_labels = [ 'chan%d' % ( i + 1 ) for i in range( bcistreams[0].nchan ) ]
    if 'ChannelNames' in bcistreams[0].params.keys():
        ch_names = bcistreams[0].params[ 'ChannelNames' ]
        if len( ch_names ) != 0:
            eeg_labels = ch_names
    for idx, label in enumerate( eeg_labels ):
        if label in labeldict.keys():
            eeg_labels[ idx ] = labeldict[ label ]
    eeg_offsets = bcistreams[0].offsets.astype( 'int32' )

    # Determine the number of samples across all files and create the groups
    nsamp = np.sum( [ dat.samples() for dat in bcistreams ] )
    H5EEGDataset.create( group, nsamp, eeg_labels, eeg_offsets, bcistreams[0].gains, 
        name = 'raw', rate = bcistreams[0].samplingrate(), bytes_per_sample = bcistreams[0].bytesperchannel )
    aux_labels = list( bcistreams[0].statedefs.keys() )
    H5EEGAuxDataset.create( group, nsamp, labels = aux_labels, 
        rate = bcistreams[0].samplingrate() )
    H5EEGEvents.create( group )

    # Read the data into the h5f file in blocks of 1 second each
    cur_samp = 0
    for dat in bcistreams:
        dat.seek( 0 )
        eeg_dset = group.eeg().dataset
        aux_dset = group.aux().dataset
        print( 'Appending %s to %s...' % ( dat.filename, h5filename ) )

        # TODO Restore pbar to its former glory
        # pbar = ProgressTracker( dat.samples(), True )
        while dat.tell() != dat.samples():
            read_block = int( dat.samplingrate() )
            signal, states = dat.decode( nsamp = read_block, apply_gains = False )
            read_block = signal.shape[1]

            eeg_dset[ cur_samp:( cur_samp + read_block ), : ] = signal.T
            for idx, label in enumerate( aux_labels ):
                try: aux_dset[ cur_samp:( cur_samp + read_block ), idx ] = np.squeeze( states[ label ] )
                except KeyError: continue                    
            cur_samp += read_block

            # TODO Restore pbar to its former glory
            # pbar.step( read_block )

    # ProgressTracker doesn't add a newline at the end
    print( '' )

    # Populate the events
    events = group.events()

    if add_everything:
        # Auto-add all events that aren't SourceTime or StimulusTime
        lame_states = ['SourceTime', 'StimulusTime']
        for idx, state in enumerate( aux_labels ):
            if state in lame_states:
                continue

            # Check our DC channel exclusion
            if no_dc:
                if state.find( 'DC' ) == 0:
                    continue

            state_data = group.aux()[ :, state ]
            t = [ idx for idx, val in enumerate( np.diff( state_data ) ) if val != 0 ]

            if len( t ) == 0:
                print( 'No events to add for state changes in %s.' % state )
                continue

            print( 'Adding %d events from state changes in %s...' % ( len( t ), state ) )

            for idx, duration in enumerate( np.diff( t ) ):
                name = '%s_%s' % ( state, state_data[ t[ idx ] + 1 ] )
                events.add_event( name, t[ idx ] + 1, duration )
            name = '%s_%s' % ( state, state_data[ t[ -1 ] + 1 ] )
            events.add_event( name, t[ -1 ] + 1, len( state_data ) - t[ -1 ] - 1 )

        if len( events ) != 0:
            print( 'Saving %d total events...' % len( events ) )
            events.write()

    else:
        # Interactive
        while True:
            print( '\n---------------------------------------------------------' )
            print( 'Currently, there are %d events...' % len( events ) )
            print( "[0] Don't add events" )
            for idx, label in enumerate( aux_labels ):
                print( '[%d] %s' % ( idx + 1, label ) )
            state_idx = raw_input( 'Parse events from state: [0] ' )
            if state_idx == "" or state_idx == "0": break

            state = aux_labels[ int( state_idx ) - 1 ]
            print( "Parsing events from state changes in %s..." % state )

            state_data = group.aux()[ :, state ]
            t = [ idx for idx, val in enumerate( np.diff( state_data ) ) if val != 0 ]
            addEvents = raw_input( "Add %d events? [y/N] " % len( t ) )
            print(addEvents)
            if addEvents.lower() == "y":
                for idx, duration in enumerate( np.diff( t ) ):
                    name = '%s_%s' % ( state, state_data[ t[ idx ] + 1 ] )
                    events.add_event( name, t[ idx ] + 1, duration )
                name = '%s_%s' % ( state, state_data[ t[ -1 ] + 1 ] )
                events.add_event( name, t[ -1 ] + 1, len( state_data ) - t[ -1 ] - 1 )

        if len( events ) != 0:
            if raw_input( "Save %d events? [y/N] " % len( events ) ).lower() == 'y':
                events.write()
    
    print( 'Done.' )

    outfile.flush()
    return outfile

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description = 'Convert BCI2000 files into H5EEG')
#     parser.add_argument('datafolder', help = 'Folder containing all of the data for the tasks', default='../Data')
#     parser.add_argument('taskname', help = 'Name of task run to convert', nargs=1)
#     parser.add_argument('-o', '--output', help = 'Set output folder for all converted files')
#     parser.add_argument( '-f', '--force', action = 'store_true', help = 'Force overwrite if file exists' )
#     parser.add_argument( '--relabel', help = 'Specify a csv (old,new) to relabel channel names' )
#     parser.add_argument( '--everything', action = 'store_true', help = 'Add all events except SourceTime' )
#     parser.add_argument( '--nodc', action = 'store_true', help = 'Exclude states beginning with "DC" from --everything' )
#     args = parser.parse_args()

#     overwrite = (args.force != None)

#     taskname = args.taskname[0]

#     matching_regex = f"{args.taskname}.*\.dat"

#     print(taskname)
#     print(args.datafolder)

#     print(args)

#     for path, subdirs, files in os.walk(args.datafolder):
#         for name in tqdm.tqdm(files):
#             if bool(re.search(f"{taskname}.*\.dat", name)):
#                 input_filename = os.path.join(path, name)
#                 output_filename = os.path.join('Data/HDF5/' + taskname, name.split('.')[0] + '.hdf5')

#                 print(input_filename)
#                 print(output_filename)

#                 h5file = convert_dat(bcistream(input_filename) , h5filename = output_filename,
#         relabelcsv = args.relabel, overwrite = overwrite, add_everything = args.everything, no_dc = args.nodc )

#                 if h5file != None: h5file.close()