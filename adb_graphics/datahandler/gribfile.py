# pylint: disable=invalid-name,too-few-public-methods

'''
Classes that load grib files.
'''

from functools import lru_cache

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
    def free_fcst_names(ds, fcst_type):

        ''' Given an opened dataset, return a dict of original variable names
        (key) and the desired name (value) '''

        ret = {}

        special_suffixes = ['max', 'min', 'acc', 'avg']
        for var in ds.variables:
            suffix = var.split('_')[-1]

            # Keeping lists of misbehaving "accumulated variables here because
            # there doesn't seem to be another way to know....

            if fcst_type == '01fcst':
                # Don't rename these variables at early hours
                odd_variables = [
                    'ASNOW',
                    'CDLYR',
                    'FRZR',
                    'LRGHR',
                    'TCDC',
                    ]
                needs_renaming = var.split('_')[0] not in odd_variables
                if suffix in special_suffixes and needs_renaming:
                    new_suffix = f'{suffix}1h'
                    ret[var] = var.replace(suffix, new_suffix)
            else:
                # Only rename these variables at late hours
                odd_variables = [
                    'APCP',
                    'CDLYR',
                    'FROZR',
                    'LRGHR',
                    'TCDC',
                    'WEASD',
                    ]
                needs_renaming = var.split('_')[0] in odd_variables
                contains_suffix = [suf for suf in special_suffixes if suf in
                                   suffix and '1h' not in suffix]
                if contains_suffix and needs_renaming:
                    ret[var] = var.replace(suffix, contains_suffix[0])

        return ret

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

        # 0h and 1h accumulated forecast variables are named differently than
        # the rest of the forecast hours. Rename those accumulated variables if
        # needed.
        for fcst_type in ['01fcst', 'free_fcst']:

            if filenames.get(fcst_type):
                for filename in filenames.get(fcst_type):
                    print(f'Loading grib2 file: {fcst_type}, {filename}')

                # Rename variables to match free forecast variables
                dataset = xr.open_mfdataset(
                    filenames[fcst_type],
                    **self.open_kwargs,
                    )

                renaming = self.free_fcst_names(dataset, fcst_type)
                if renaming:
                    print(f'RENAMING VARIABLES:')
                for old_name, new_name in renaming.items():
                    print(f'  {old_name:>30s}  -> {new_name}')
                dataset = dataset.rename_vars(renaming)

                if len(all_leads) == 1:
                    # Check that specific variables exist in the xarray that is
                    # already loaded (presumably 0hr), and add them if they
                    # don't. This implementation is relying on pointers to
                    # update "in place"
                    og_ds = all_leads[0]
                    bad_vars = [
                        'APCP_P8_L1_{grid}_acc',
                        'ACPCP_P8_L1_{grid}_acc',
                        'FROZR_P8_L1_{grid}_acc',
                        'NCPCP_P8_L1_{grid}_acc',
                        'WEASD_P8_L1_{grid}_acc',
                        ]
                    bad_vars = [v.format(grid=self.grid_suffix) for v in \
                            bad_vars]
                    for bad_var in bad_vars:
                        # Check to see if the bad variable is in the current
                        # dataset and NOT in the original dataset.
                        if bad_var not in og_ds.variables and \
                            dataset.get(bad_var) is not None:
                            print(f'Adding {bad_var} to og ds')
                            # Dupplicate the accumulated variable with the
                            # required name
                            og_ds[bad_var] = og_ds.get(f'{bad_var}1h')
                all_leads.append(dataset)

        ret = xr.combine_nested(all_leads,
                                compat='override',
                                concat_dim=list(self.coord_dims.keys())[0],
                                coords='minimal',
                                data_vars='all',
                                )
        return ret

    @property
    @lru_cache()
    def open_kwargs(self):

        ''' Defines the key word arguments used by the various calls to XArray
        open_mfdataset '''

        return dict(
            backend_kwargs=dict(format="grib2"),
            combine='nested',
            compat='override',
            concat_dim=list(self.coord_dims.keys())[0],
            coords='minimal',
            engine='pynio',
            lock=False,
            )
