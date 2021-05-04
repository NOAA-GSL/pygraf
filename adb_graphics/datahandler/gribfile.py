# pylint: disable=invalid-name,too-few-public-methods

'''
Classes that load grib files.
'''

import xarray as xr

class GribFile():

    ''' Wrappers and helper functions for interfacing with pyNIO.'''

    def __init__(self, filename, **kwargs):

        # pylint: disable=unused-argument

        self.filename = filename
        self.contents = self._load()

    def _load(self):

        ''' Internal method that opens the grib file. Returns a grib message
        iterator. '''

        return xr.open_dataset(self.filename,
                               engine='pynio',
                               lock=False,
                               backend_kwargs=dict(format="grib2"),
                               )

class GribFiles():

    ''' Class for loading in a set of grib files and combining them over
    forecast hours. '''

    def __init__(self, coord_dims, filenames, filetype, **kwargs):

        '''
        Arguments:

          coord_dims  dict containing the name of the dimension to
                      concat (key), and a list of its values (value).
                        Ex: {'fhr': [2, 3, 4]}
          filenames   dict containing list of files names for the 0h and 1h
                      forecast lead times ('01fcst'), and all the free forecast
                      hours after that ('free_fcst').
          filetype    key to use for dict when setting variable_names

        Keyword Arguments:
          model       string describing the model type
        '''

        self.model = kwargs.get('model', '')

        self.filenames = filenames
        self.filetype = filetype
        self.coord_dims = coord_dims
        self.grid_suffix = self._get_grid_suffix(filenames)
        self.contents = self._load()


    def append(self, filenames):

        ''' Add a single new slice to existing data set. Must match coord_dims
        and filetype of original dataset. Updates current contents of Object'''

        self.contents = self._load(filenames)

    @staticmethod
    def _get_grid_suffix(filenames):

        ''' Return the suffix of the first variable with 4 sections (split on _)
        in the file. This should correspond to the grid tag. '''

        for files in filenames.values():
            if files:
                gfile = xr.open_dataset(files[0],
                                        engine='pynio',
                                        lock=False,
                                        backend_kwargs=dict(format="grib2"),
                                        )
                for var in gfile.keys():
                    vsplit = var.split('_')
                    if len(vsplit) == 4:
                        gfile.close()
                        return vsplit[-1]
        return 'GRID NOT FOUND'

    def _load(self, filenames=None):

        ''' Load the set of files into a single XArray structure. '''

        all_leads = [] if filenames is None else [self.contents]
        filenames = self.filenames if filenames is None else filenames
        var_names = self.variable_names

        # 0h and 1h forecast variables are named like the keys in var_names,
        # while the free forecast hours are named like the values in var_names.
        # Need to do a bit of cleanup to get them into a single datastructure.
        names = {
            '01fcst': var_names,
            'free_fcst': var_names.values(),
            }

        for fcst_type, vnames in names.items():

            if filenames.get(fcst_type):
                for filename in filenames.get(fcst_type):
                    print(f'Loading grib2 file: {fcst_type}, {filename}')

                if fcst_type == '01fcst':
                    if filenames.get(fcst_type):
                        # Rename variables to match free forecast variables
                        all_leads.append(xr.open_mfdataset(
                            filenames[fcst_type],
                            **self.open_kwargs(vnames),
                            ).rename_vars(vnames))
                else:
                    all_leads.append(xr.open_mfdataset(
                        filenames[fcst_type],
                        **self.open_kwargs(vnames),
                        ))

        ret = xr.combine_nested(all_leads,
                                compat='override',
                                concat_dim=list(self.coord_dims.keys())[0],
                                coords='minimal',
                                data_vars='minimal',
                                )

        return ret

    def open_kwargs(self, vnames):

        ''' Defines the key word arguments used by the various calls to XArray
        open_mfdataset '''

        return dict(
            backend_kwargs=dict(format="grib2"),
            combine='nested',
            compat='override',
            concat_dim=list(self.coord_dims.keys())[0],
            coords='minimal',
            data_vars=vnames,
            engine='pynio',
            lock=False,
            )

    @property
    def variable_names(self):

        '''
        Defines the variable name transitions that need to happen for each
        model to combine along forecast hours.

        Keys are original variable names, values are the updated variable names.

        Choosing to update the 0 and 1 hour forecast variables to the longer
        lead times for efficiency's sake.
        '''

        grid = self.grid_suffix
        names = {
            'hrrr': {
                f'REFC_P0_L10_{grid}': f'REFC_P0_L10_{grid}',
                f'MXUPHL_P8_2L103_{grid}_max': f'MXUPHL_P8_2L103_{grid}_max1h',
                f'UGRD_P0_L103_{grid}': f'UGRD_P0_L103_{grid}',
                f'VGRD_P0_L103_{grid}': f'VGRD_P0_L103_{grid}',
                f'WEASD_P8_L1_{grid}_acc': f'WEASD_P8_L1_{grid}_acc1h',
                f'APCP_P8_L1_{grid}_acc': f'APCP_P8_L1_{grid}_acc1h',
                f'PRES_P0_L1_{grid}': f'PRES_P0_L1_{grid}',
                f'VAR_0_7_200_P8_2L103_{grid}_min': f'VAR_0_7_200_P8_2L103_{grid}_min1h',
                },
            'rap': {
                f'REFC_P0_L10_{grid}': f'REFC_P0_L10_{grid}',
                f'UGRD_P0_L103_{grid}': f'UGRD_P0_L103_{grid}',
                f'VGRD_P0_L103_{grid}': f'VGRD_P0_L103_{grid}',
                f'WEASD_P8_L1_{grid}_acc': f'WEASD_P8_L1_{grid}_acc1h',
                f'APCP_P8_L1_{grid}_acc': f'APCP_P8_L1_{grid}_acc1h',
                f'PRES_P0_L1_{grid}': f'PRES_P0_L1_{grid}',
                },
            'rtma': {
                },
            'rrfs': {
                f'REFC_P0_L200_{grid}': f'REFC_P0_L200_{grid}',
                f'MXUPHL_P8_2L103_{grid}_max': f'MXUPHL_P8_2L103_{grid}_max1h',
                f'UGRD_P0_L103_{grid}': f'UGRD_P0_L103_{grid}',
                f'VGRD_P0_L103_{grid}': f'VGRD_P0_L103_{grid}',
                f'WEASD_P8_L1_{grid}_acc': f'WEASD_P8_L1_{grid}_acc1h',
                f'APCP_P8_L1_{grid}_acc': f'APCP_P8_L1_{grid}_acc1h',
                f'PRES_P0_L1_{grid}': f'PRES_P0_L1_{grid}',
                f'VAR_0_7_200_P8_2L103_{grid}_min': f'VAR_0_7_200_P8_2L103_{grid}_min1h',
                },
            }

        return names[self.model]
