import numpy as np
import pymorph
from skimage import measure


# Find the columns to the left and right of the centre in which the maximum
# gradient magnitude of a profile is to be found.  Because PROF should
# represent a 'hill-like' vessel, the potential edge on the left has the
# largest positive gradient, while the potential edge on the right has
# largest negative gradient.  REGION_LENGTH defines the size of the search
# region around the centre that is used when looking for the gradient
# locations.
def find_maximum_gradient_columns(profile, width_estimate): #this profile is the mean profile, shape [20,]

    col_central = len(profile) //2
    region_length = np.int(np.ceil(width_estimate) + 1)
    if region_length >= col_central:
        region_length = col_central

    profile[1:-1] = profile[2:] - profile[:-2]
    profile[:col_central - region_length] = np.nan
    profile[col_central+region_length:] = np.nan

    leftCol = np.nanargmax(profile[:col_central])
    rightCol = np.nanargmin(profile[col_central:])

    rightCol = rightCol + col_central

    return leftCol, rightCol



def gaussian_filter_1d_sigma(sigma):
    if sigma <= 0:
        return 1
    else:
        len = np.int(np.ceil(sigma*3) *2 +1)
        xx = np.arange(0, len)
        xx = xx - xx[-1] / 2
        g = np.exp(-xx**2 / (2*sigma ** 2))
        g = g / np.sum(g)

        return g



def compute_discrete_2d_derivative(profile):  #this profile is the profile matrix, shape [n, 20]
    profile1 = np.concatenate((profile[:, 0].reshape(-1, 1), profile), axis=1)
    profile2 = np.concatenate((profile, profile[:, -1].reshape(-1, 1)), axis=1)
    profile_2d = profile1[:, :-1] + profile2[:, 1:] - 2*profile

    return profile_2d



#################################################################################
#% Find the closest (non-NaN) value in CROSSINGS immediately after to a given column.
# COLUMN should be a scalar, or a vector with a length SIZE(CROSSINGS, 1)
def find_next_crossing(crossings, column):
    crossings0 = crossings.copy()
    crossings0[crossings0 < column] = np.NAN
    cross_next = np.nanmin(crossings0, 1)
    return cross_next


def find_previous_crossing(crossings, column):
    crossings0 = crossings.copy()
    crossings0[crossings0 > column] = np.NAN
    cross_prev = np.nanmax(crossings0, 1)
    return cross_prev

#For each row in CROSSINGS, find the value closest to that of COLUMN that
#is either lower (if SEARCH_PREVIOUS == TRUE) or higher (if
#SEARCH_PREVIOUS == FALSE).
def find_closest_crossing(crossings, column, search_previous):
    search_next = not search_previous
    cross = np.NAN(crossings.shape[0])
    if np.any(search_previous):
        cross[search_previous] = find_previous_crossing(crossings[search_previous,:], column)
    if np.any(search_next):
        cross[search_next] = find_next_crossing(crossings[search_next,:], column)

    return cross


# For a matrix CROSSINGS containing crossing values in (approximately) the
# correct column and NaNs elsewhere, look for the connected components of
# the non-NaN regions in order to identify potential connected edges.
# Exclude those that do not come within REGION_LENGTH pixels of COLUMN at
# any location, then retain only the CROSSINGS that are part of the largest
# remaining potential edge regions for the corresponding row (i.e. set
# everything else to NaN).
# The idea is that long sequences of zero-crossings that are close to one
# another are more likely to be parts of true vessel edges than short or
# isolated zero-crossings.  In the idea case, there would be a single trail
# of zero-crossings to the left and right of centre for each vessel.
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)   #not print the warning
def find_most_connected_crossings(crossings, column, region_length):

    ###may cause a problem in here if column is not scalar
    # if np.isscalar(column):
    #     bw_region = np.abs(crossings-column) <= region_length
    # else:
    #     bw_region = np.abs()

    crossings0 = crossings.copy()

    BW_region = np.array(np.abs(crossings0 - column) <= region_length, dtype=bool)
    ##warning: RuntimeWarning: invalid value encountered in absolute

    Search_region = pymorph.cdilate(pymorph.binary(BW_region), pymorph.binary(np.isfinite(crossings0)))
    Search_region = np.array(Search_region, dtype=bool)

    crossings0[np.bitwise_not(Search_region)] = np.NAN

    finite_crossing = np.isfinite(crossings0)
    if np.sum(np.isfinite(crossings0), 1).all() <= 1:
        return crossings0

    ##Label each connected trail of crossings
    Cross_label = measure.label(finite_crossing)
    Cross_LabelNum = np.zeros(Cross_label.shape)
    if np.max(Cross_label) == 1:
        crossings0[Cross_label==1] = np.NAN
    else:
        #count the number of crossings per trail
        for templabel in xrange(1, np.max(Cross_label) +1):
            Cross_LabelNum[Cross_label==templabel] = np.count_nonzero(Cross_label[Cross_label==templabel])

    BW_region = Cross_LabelNum == np.tile(np.max(Cross_LabelNum, 1), (1, Cross_LabelNum.shape[1]))

    crossings0[np.bitwise_not(BW_region)] = np.NAN

    return crossings0



# A custom linear interpolation method for extracting edge coordinates.  It
# assumes that ROWS gives the rows inside IM_PROFILES_ROWS/COLS, and will be
# valid integer values.  COLS gives the columns in the same matrices, and
# will be within range but potentially non-integer, and so interpolation
# will be used.

def get_side(im_profiles_rows, im_profiles_cols, rows, cols):
    cols_floor = np.int16(np.floor(cols))
    cols_diff = cols - cols_floor

    # inds_floor = sub2ind2d(im_profiles_rows.shape, rows, cols_floor)
    # ins_floor_plus = inds_floor + im_profiles_rows.shape[0]
    side_rows = im_profiles_rows[rows, cols_floor] * (1-cols_diff) + im_profiles_rows[rows, cols_floor+1]*cols_diff
    side_cols = im_profiles_cols[rows, cols_floor+1] * (1-cols_diff) + im_profiles_cols[rows, cols_floor+1]*cols_diff
    side = (side_rows, side_cols)
    return side



# def sub2ind2d(siz, r, c):
#     index = r + siz[0]*(c-1)
#     return index