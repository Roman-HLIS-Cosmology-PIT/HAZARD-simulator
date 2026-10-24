import numpy as np


class MaterialClass:
    """
    Material property container and calculator for Hg(1−x)Cd(x)Te detector media.

    This class provides elemental constants and derived bulk properties for the
    HgCdTe alloy used in infrared detectors, including:
      - Mean excitation energy via Bragg’s logarithmic rule
      - Radiation length using PDG approximations and mixture rules
      - Mass density assuming a zincblende crystal structure with Vegard’s law
      - Number-averaged effective atomic number (Z) and atomic mass (A)

    These quantities are used in energy-loss modeling, charge transport, and
    radiation-interaction calculations within GCRsim.

    Parameters
    ----------
    I_Hg : float, optional
        Mean excitation energy of mercury (Hg).
        Units: electronvolts (eV). Default is 800.0 eV.
    I_Cd : float, optional
        Mean excitation energy of cadmium (Cd).
        Units: electronvolts (eV). Default is 469.0 eV.
    I_Te : float, optional
        Mean excitation energy of tellurium (Te).
        Units: electronvolts (eV). Default is 485.0 eV.

    Attributes
    ----------
    I_Hg : float
        Mean excitation energy of Hg (eV).
    I_Cd : float
        Mean excitation energy of Cd (eV).
    I_Te : float
        Mean excitation energy of Te (eV).

    Z_Hg : int
        Atomic number of Hg (number of electrons per atom). Units: dimensionless.
    Z_Cd : int
        Atomic number of Cd. Units: dimensionless.
    Z_Te : int
        Atomic number of Te. Units: dimensionless.

    A_Hg : float
        Atomic mass of Hg. Units: g/mol.
    A_Cd : float
        Atomic mass of Cd. Units: g/mol.
    A_Te : float
        Atomic mass of Te. Units: g/mol.

    Notes
    -----
    - Composition is parameterized by ``x``, the molar fraction of Cd in Hg(1−x)Cd(x)Te.
    - The zincblende crystal structure is assumed with 4 formula units per unit cell.
    - Lattice constants are interpolated using Vegard’s law.
    - Radiation length calculations follow PDG mixture rules.
    - Mean excitation energies follow Bragg’s logarithmic additivity rule.

    References
    ----------
    - NIST STAR database for elemental mean excitation energies:
      https://physics.nist.gov/PhysRefData/Star/Text/method.html
    - Particle Data Group (PDG), Review of Particle Physics: Radiation Lengths.
    """

    def __init__(self, I_Hg=800.0, I_Cd=469.0, I_Te=485.0):
        # Creating a class for the material data of HgCdTe detector
        # Mean excitation energies for the elements (in eV), data taken from https://physics.nist.gov/PhysRefData/Star/Text/method.html
        self.I_Hg = I_Hg  # in eV, from NIST: https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=080
        self.I_Cd = I_Cd  # in eV, from NIST: https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=048
        self.I_Te = I_Te  # in eV, from NIST: https://pml.nist.gov/cgi-bin/Star/compos.pl?matno=052
        # put into the argument

        # Atomic numbers (number of electrons per atom) -- perchance redundant
        self.Z_Hg = 80
        self.Z_Cd = 48
        self.Z_Te = 52

        # Atomic masses (g/mol) for each element:
        # Mercury (Hg)
        self.A_Hg = 200.59
        # Cadmium (Cd)
        self.A_Cd = 112.41
        # Tellurium (Te)
        self.A_Te = 127.60

    def mean_excitation_energy_HgCdTe(self, x):
        """
        Calculate the mean excitation energy for Hg_(1−x)Cd_(x)Te using Bragg's sum rule.

        Parameters
        ----------
        x : float
            Fractional composition of Cd in the alloy (0 ≤ x ≤ 1).
            The Hg fraction is automatically taken as (1 − x).

        Returns
        -------
        float
            Effective mean excitation energy of the compound in electronvolts (eV),
            computed via Bragg’s logarithmic rule from the elemental contributions.
        """

        # Electrons contributed by each element in the formula unit Hg_(1-x)Cd_(x)Te
        electrons_Hg = (1 - x) * self.Z_Hg
        electrons_Cd = x * self.Z_Cd
        electrons_Te = self.Z_Te

        # Total number of electrons in the formula unit
        total_electrons = electrons_Hg + electrons_Cd + electrons_Te

        # Weighting factors based on electron contribution
        w_Hg = electrons_Hg / total_electrons
        w_Cd = electrons_Cd / total_electrons
        w_Te = electrons_Te / total_electrons

        # Compute the logarithmic average (Bragg's rule):
        lnI = w_Hg * np.log(self.I_Hg) + w_Cd * np.log(self.I_Cd) + w_Te * np.log(self.I_Te)
        I_compound = np.exp(lnI)

        return I_compound

    def radiation_length_HgCdTe(self, x):
        """
        Compute the radiation length (in g/cm^2) for Hg(1-x)Cd(x)Te.

        Uses the PDG approximate formula for the radiation length of an element:

            X0 = 716.4 * A / (Z*(Z+1)*ln(287/sqrt(Z)))   [g/cm^2]

        and for a compound:

            1/X0_compound = sum_i (w_i / X0_i)

        where w_i = (N_i * A_i) / (sum_j N_j * A_j) are the weight fractions.

        Parameters
        ----------
        x : float
            Molar fraction of Cd (and thus Hg molar fraction is 1-x).

        Returns
        -------
        X0_compound : float
            Radiation length of the compound in g/cm^2.
        """

        # Helper function: Radiation length for an element (in g/cm^2)
        def X0_element(Z, A):
            return 716.4 * A / (Z * (Z + 1) * np.log(287 / np.sqrt(Z)))

        # Compute radiation lengths for individual elements:
        X0_Hg = X0_element(self.Z_Hg, self.A_Hg)
        X0_Cd = X0_element(self.Z_Cd, self.A_Cd)
        X0_Te = X0_element(self.Z_Te, self.A_Te)

        # Molar amounts: Hg: (1-x), Cd: x, Te: 1.
        # Total molar mass of the compound:
        A_tot = (1 - x) * self.A_Hg + x * self.A_Cd + self.A_Te

        # Weight fractions:
        w_Hg = (1 - x) * self.A_Hg / A_tot
        w_Cd = x * self.A_Cd / A_tot
        w_Te = self.A_Te / A_tot

        # Radiation length of the compound (in g/cm^2):
        X0_compound = 1.0 / (w_Hg / X0_Hg + w_Cd / X0_Cd + w_Te / X0_Te)

        return X0_compound

    def density_HgCdTe(self, x):
        """
        Compute the density (in g/cm^3) of Hg(1-x)Cd(x)Te.

        Assumes:
          - Formula unit: 1 cation (Hg with fraction 1-x or Cd with fraction x) + 1 Te.
          - Zincblende crystal structure (4 formula units per unit cell).
          - Vegard's law for the lattice constant.

        Parameters
        ----------
        x : float
            Molar fraction of Cd (and thus Hg molar fraction is 1-x).

        Returns
        -------
        density : float
            Density of Hg(1-x)Cd(x)Te in g/cm^3.
        """

        # Molar mass of the compound (g/mol)
        M = (1 - x) * self.A_Hg + x * self.A_Cd + self.A_Te

        # Lattice constants (in cm) - 1 Å = 1e-8 cm
        a_HgTe = 6.46e-8  # HgTe lattice constant in cm
        a_CdTe = 6.48e-8  # CdTe lattice constant in cm

        # Vegard's law: linear interpolation of lattice constant
        a = (1 - x) * a_HgTe + x * a_CdTe

        # For zincblende structure: 4 formula units per unit cell
        # Volume per formula unit = a^3 / 4
        volume_per_formula = a**3 / 4

        # Avogadro's number (mol^-1)
        N_A = 6.02214076e23

        # Mass per formula unit in grams
        mass_per_formula = M / N_A

        # Density in g/cm^3
        density = mass_per_formula / volume_per_formula

        return density

    def mean_Z_A_HgCdTe(self, x):
        """
        Compute the number-averaged mean atomic number (Z) and atomic mass (A)
        for Hg(1-x)Cd(x)Te in a zincblende structure.

        Parameters:
        -----------
        x : float
            Molar fraction of Cd (and hence Hg fraction is 1-x).

        Returns:
        --------
        Z_mean : float
            Number-averaged mean atomic number.
        A_mean : float
            Number-averaged mean atomic mass (in g/mol).
        """

        # There are (1-x) moles of Hg, x moles of Cd, and 1 mole of Te per formula unit.
        # Total number of atoms per formula unit = (1-x) + x + 1 = 2.
        total_atoms = 2

        Z_mean = ((1 - x) * self.Z_Hg + x * self.Z_Cd + self.Z_Te) / total_atoms
        A_mean = ((1 - x) * self.A_Hg + x * self.A_Cd + self.A_Te) / total_atoms

        return Z_mean, A_mean
