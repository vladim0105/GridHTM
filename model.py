from typing import TypedDict, NamedTuple

import htm.algorithms
import htm.algorithms as algos
import numpy as np
from cv2 import cv2
from htm.bindings.sdr import SDR

import utils


def grid_mean_aggr_func(anoms, axis=None):
    anoms[anoms==0] = np.nan
    mean = np.nanmean(anoms, axis=axis)
    mean[np.isnan(mean)] = 0
    print(np.isnan(mean).any())
    return mean


class SpatialPoolerArgs:
    def __init__(self):
        self.inputDimensions = (0,)
        self.columnDimensions = (0,)
        self.potentialPct = 0.01
        """The percent of the inputs, within a column's
            potential radius, that a column can be connected to. If set to
            1, the column will be connected to every input within its
            potential radius. This parameter is used to give each column a
            unique potential pool when a large potentialRadius causes
            overlap between the columns. At initialization time we choose
            ((2*potentialRadius + 1)^(# inputDimensions) * potentialPct)
            input bits to comprise the column's potential pool."""
        self.potentialRadius = 100
        """This parameter determines the extent of the
            input that each column can potentially be connected to. This
            can be thought of as the input bits that are visible to each
            column, or a 'receptive field' of the field of vision. A large
            enough value will result in global coverage, meaning
            that each column can potentially be connected to every input
            bit. This parameter defines a square (or hyper square) area: a
            column will have a max square potential pool with sides of
            length (2 * potentialRadius + 1)."""
        self.globalInhibition = True
        """If true, then during inhibition phase the
            winning columns are selected as the most active columns from the
            region as a whole. Otherwise, the winning columns are selected
            with respect to their local neighborhoods. Global inhibition
            boosts performance significantly but there is no topology at the
            output."""

        self.localAreaDensity = 0.01
        """The desired density of active columns within
            a local inhibition area (the size of which is set by the
            internally calculated inhibitionRadius, which is in turn
            determined from the average size of the connected potential
            pools of all columns). The inhibition logic will insure that at
            most N columns remain ON within a local inhibition area, where
            N = localAreaDensity * (total number of columns in inhibition
            area). """
        self.synPermInactiveDec = 0.08
        """The amount by which the permanence of an
            inactive synapse is decremented in each learning step."""
        self.synPermActiveInc = 0.14
        """The amount by which the permanence of an
            active synapse is incremented in each round."""
        self.synPermConnected = 0.5
        self.boostStrength = 0.0
        """A number greater or equal than 0, used to
            control boosting strength. No boosting is applied if it is set to 0.
            The strength of boosting increases as a function of boostStrength.
            Boosting encourages columns to have similar activeDutyCycles as their
            neighbors, which will lead to more efficient use of columns. However,
            too much boosting may also lead to instability of SP outputs."""
        self.stimulusThreshold = 1
        self.wrapAround = False
        self.dutyCyclePeriod = 1000
        """The period used to calculate duty cycles.
            Higher values make it take longer to respond to changes in
            boost. Shorter values make it potentially more unstable and
            likely to oscillate."""
        self.minPctOverlapDutyCycle = 0.001
        """A number between 0 and 1.0, used to set
            a floor on how often a column should have at least
            stimulusThreshold active inputs. Periodically, each column looks
            at the overlap duty cycle of all other column within its
            inhibition radius and sets its own internal minimal acceptable
            duty cycle to: minPctDutyCycleBeforeInh * max(other columns'
            duty cycles). On each iteration, any column whose overlap duty
            cycle falls below this computed value will get all of its
            permanence values boosted up by synPermActiveInc. Raising all
            permanences in response to a sub-par duty cycle before
            inhibition allows a cell to search for new inputs when either
            its previously learned inputs are no longer ever active, or when
            the vast majority of them have been "hijacked" by other columns."""
        self.seed = 0


class TemporalMemoryArgs:
    def __init__(self):
        self.columnDimensions = (0,)
        self.seed = 0

        self.initialPermanence = 0.21
        """Initial permanence of a new synapse."""

        self.predictedSegmentDecrement = 0.01
        self.connectedPermanence = 0.7

        self.permanenceIncrement = 0.01
        """Amount by which permanences of synapses are incremented during learning."""

        self.permanenceDecrement = 0.01
        self.minThreshold = 10
        """If the number of potential synapses active on a segment is at least
            this threshold, it is said to be "matching" and is eligible for
            learning."""
        self.activationThreshold = 13
        """
        If the number of active connected synapses on a segment is at least
        this threshold, the segment is actived.
        """
        self.cellsPerColumn = 16
        """
        Number of cells per mini-column
        """
        self.maxNewSynapseCount = 20
        """
        The maximum number of synapses added to a segment during learning.
        """


class SpatialPooler:
    def __init__(self, sp_args: SpatialPoolerArgs):
        self.sp = algos.SpatialPooler(**sp_args.__dict__)
        self.num_active_columns = round(self.sp.getNumColumns() * self.sp.getLocalAreaDensity())

    def __call__(self, encoded_sdr: SDR, learn) -> SDR:
        active_sdr = SDR(self.sp.getColumnDimensions())
        # Run the spatial pooler
        self.sp.compute(input=encoded_sdr, learn=learn, output=active_sdr)
        return active_sdr


class TemporalMemory:
    def __init__(self, tm_args: TemporalMemoryArgs):
        self.tm = algos.TemporalMemory(**tm_args.__dict__)

    def __call__(self, active_sdr: SDR, learn):
        # Calculate predictive cells by activating dendrites by themselves, does not affect output
        #self.tm.activateDendrites(learn=False)
        #n_pred_cells = self.tm.getPredictiveCells().getSum()

        self.tm.compute(active_sdr, learn)
        # Extract the predicted SDR and convert it to a tensor
        predicted = self.tm.getActiveCells()
        # Extract the anomaly score
        anomaly = self.tm.anomaly
        n_pred_cells = 0
        return predicted, anomaly, n_pred_cells


class GridHTM:
    def __init__(self, frame_shape, sp_grid_size, tm_grid_size, sp_args: SpatialPoolerArgs, tm_args: TemporalMemoryArgs,
                 min_sparsity=1, sparsity=15, temporal_size=1, aggr_func=grid_mean_aggr_func):
        assert sp_grid_size == sp_args.inputDimensions[0], "SP grid size and SP input dimensions must match!"
        assert tm_grid_size == tm_args.columnDimensions[0] == sp_args.columnDimensions[
            0], "TM grid size and SP/TM column dimensions must match!"
        assert sp_grid_size % tm_grid_size == 0, "SP Grid size must be divisible by TM Grid side!"
        assert temporal_size > 0, "Temporal size must be larger than 0!"
        np.random.seed(sp_args.seed) # Sets the seed to be used by numpy, mainly used for empty pattern generation
        self.input_shape = frame_shape
        self.prev_input = np.ones(shape=self.input_shape)
        self.sp_grid_size = sp_grid_size
        self.tm_grid_size = tm_grid_size
        self.sp_args = sp_args
        self.tm_args = tm_args
        self.sparsity = sparsity  # How many ON bits per cell in the grid the encoding should produce
        self.min_sparsity = min_sparsity  # Minimum bits required before a cell is considered not empty
        self.empty_pattern = utils.random_bit_array(shape=(sp_grid_size, sp_grid_size), num_ones=sparsity)
        self.aggr_func = aggr_func
        self.sps = []
        self.tms = []
        self.temporal_size = temporal_size

        tm_args.columnDimensions = (tm_args.columnDimensions[0] * self.temporal_size, tm_args.columnDimensions[1])
        # Spatial Pooler Init
        for i in range(frame_shape[0] // sp_grid_size):
            sps_inner = []
            for j in range(frame_shape[1] // sp_grid_size):
                sp_args.seed += 1
                sps_inner.append(SpatialPooler(sp_args))
            self.sps.append(sps_inner)
        # Temporal Memory Init
        ratio = sp_grid_size // tm_grid_size
        for i in range(frame_shape[0] // (ratio * tm_grid_size)):
            tms_inner = []
            for j in range(frame_shape[1] // (ratio * tm_grid_size)):
                tms_inner.append(TemporalMemory(tm_args))
            self.tms.append(tms_inner)
        # Shape tm_grid_x, tm_grid_y, time, tm_grid_size, tm_grid_size
        self.prev_sp_grid_outputs = np.zeros(shape=(
            len(self.tms), len(self.tms[0]), self.temporal_size,
            self.tm_grid_size, self.tm_grid_size))

    def grid_sp(self, sp_input: np.ndarray):
        sp_output = np.zeros(shape=(self.tm_grid_size * len(self.sps), self.tm_grid_size * len(self.sps[0])))
        for i in range(len(self.sps)):
            for j in range(len(self.sps[i])):
                sp = self.sps[i][j]
                val = sp_input[i * self.sp_grid_size: (i + 1) * self.sp_grid_size,
                      j * self.sp_grid_size: (j + 1) * self.sp_grid_size]
                # Check if empty
                if val.sum() < self.min_sparsity:
                    val = self.empty_pattern
                sdr_cell = numpy_to_sdr(val)
                sp_cell_output = sdr_to_numpy(sp(sdr_cell, learn=True))
                sp_output[i * self.tm_grid_size: (i + 1) * self.tm_grid_size,
                j * self.tm_grid_size: (j + 1) * self.tm_grid_size] = sp_cell_output
        return sp_output

    def grid_tm(self, sp_output: np.ndarray, current_input: np.ndarray, prev_input: np.ndarray):
        anoms = np.zeros(shape=(len(self.tms), len(self.tms[0])))
        colored_sdr_arr = np.zeros(shape=(sp_output.shape[0], sp_output.shape[1], 3), dtype=np.uint8)

        for i in range(len(self.tms)):
            for j in range(len(self.tms[i])):
                tm = self.tms[i][j]
                sp_grid_output = sp_output[i * self.tm_grid_size: (i + 1) * self.tm_grid_size,
                                 j * self.tm_grid_size: (j + 1) * self.tm_grid_size]
                # Shift old sp_outputs out and add the prev one. 0 is newest.
                for k in range(self.prev_sp_grid_outputs.shape[2]-1, 0, -1):
                    self.prev_sp_grid_outputs[i, j, k] = self.prev_sp_grid_outputs[i, j, k-1]
                self.prev_sp_grid_outputs[i, j, 0] = sp_grid_output
                val = sp_grid_output
                for k in range(1, self.prev_sp_grid_outputs.shape[2]):
                    val = np.concatenate((val, self.prev_sp_grid_outputs[i, j, k]), axis=0)
                sdr_cell = numpy_to_sdr(val)
                pred, anom, n_pred_cells = tm(sdr_cell, learn=True)

                # Stabilize Anomaly Score
                prev_val = prev_input[i * self.sp_grid_size: (i + 1) * self.sp_grid_size,
                           j * self.sp_grid_size: (j + 1) * self.sp_grid_size]
                current_val = current_input[i * self.sp_grid_size: (i + 1) * self.sp_grid_size,
                           j * self.sp_grid_size: (j + 1) * self.sp_grid_size]
                if (prev_val == 0).all() and (current_val == 1).any():
                    anom = 0
                colored_sdr_arr[i * self.tm_grid_size: (i + 1) * self.tm_grid_size,
                j * self.tm_grid_size: (j + 1) * self.tm_grid_size, 0] = int(
                    60 * (1 - anom))
                colored_sdr_arr[i * self.tm_grid_size: (i + 1) * self.tm_grid_size,
                j * self.tm_grid_size: (j + 1) * self.tm_grid_size, 1] = 255
                colored_sdr_arr[i * self.tm_grid_size: (i + 1) * self.tm_grid_size,
                j * self.tm_grid_size: (j + 1) * self.tm_grid_size, 2] = 255 * (
                        1 - val[:self.tm_grid_size, :self.tm_grid_size])
                anoms[i, j] = anom

        colored_sdr_arr = cv2.cvtColor(colored_sdr_arr, cv2.COLOR_HSV2BGR)
        return self.aggr_func(anoms.flatten()), colored_sdr_arr, anoms

    def __call__(self, encoded_input: np.ndarray):
        sp_output = self.grid_sp(encoded_input)
        anom_score, colored_sp_output, raw_anoms = self.grid_tm(sp_output, encoded_input, self.prev_input)
        self.prev_input = encoded_input
        return anom_score, colored_sp_output, raw_anoms


def numpy_to_sdr(arr: np.ndarray) -> SDR:
    sdr = SDR(dimensions=arr.shape)
    sdr.dense = arr.tolist()
    return sdr


def sdr_to_numpy(sdr: SDR) -> np.ndarray:
    return np.array(sdr.dense)
