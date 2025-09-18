# geocollaborate-utilities

Python utility scripts and assets to support the [GeoCollaborate](https://geocollaborate.com) platform. The main component is a C-Star Operational KMZ generator that fetches oceanographic data from ERDDAP servers and creates KMZ files with NHC weather tracking.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Dependencies
- Clone the repository and navigate to the root directory
- Install Python dependencies:
  - Required: `pip3 install -r requirements.txt` -- installs requests==2.31.0. Takes ~1 second.
  - Optional: `pip3 install matplotlib shapely` -- enables chart generation and enhanced geometry operations. Takes 15+ seconds. NEVER CANCEL: May timeout due to network issues, set timeout to 300+ seconds.
- Verify installation: `python3 -c "import requests; print('Core dependencies ready')"`
- Test script compilation: `python3 -m py_compile scripts/cstar_kml.py`

### Running the Main Script
- Primary script: `scripts/cstar_kml.py` -- C-Star Operational KMZ generator
- Help: `python3 scripts/cstar_kml.py --help`
- Basic execution (requires network): `echo "" | python3 scripts/cstar_kml.py --nhc-cone-url "" --out test_output.kmz`
- Script execution takes ~1 second when ERDDAP servers are unreachable (offline mode)
- **NETWORK DEPENDENCY**: Script requires internet access to fetch data from ERDDAP servers (data.pmel.noaa.gov, www.aoml.noaa.gov)
- When offline: Script will fail gracefully with network error messages and exit code 0

### Validation and Testing
- **NO FORMAL TEST SUITE**: Repository has no unittest, pytest, or automated test framework
- Validate syntax: `python3 -m py_compile scripts/cstar_kml.py && echo "Script compiles successfully"`
- Test basic functionality: Run the script with mock parameters as shown above
- **NO LINTING CONFIGURATION**: No flake8, pylint, or black configuration files present
- **NO BUILD SYSTEM**: Repository requires no compilation or build steps beyond Python dependency installation

## Manual Validation Scenarios

After making changes to the script:
1. **Syntax Check**: `python3 -m py_compile scripts/cstar_kml.py`
2. **Help Function**: `python3 scripts/cstar_kml.py --help` -- should display usage information
3. **Offline Mode**: `echo "" | timeout 10 python3 scripts/cstar_kml.py --nhc-cone-url "test" --out test.kmz` -- should fail gracefully with network errors
4. **Output Generation**: When network is available, verify KMZ file generation in Cstar_Locations/ directory

## Repository Structure

### Key Files and Directories
```
.
├── README.md              # Basic project description  
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies (requests only)
├── scripts/
│   └── cstar_kml.py      # Main KMZ generation script (~1200 lines)
└── Cstar_Locations/       # Default output directory
    ├── cstar_locations.kmz    # Generated KMZ file (Zip archive)
    └── _kmz_stage/           # Temporary staging directory
```

### Script Functionality
- **Input**: Fetches oceanographic data from ERDDAP servers
- **Processing**: Generates track data, charts, and geographic visualizations
- **Output**: Creates KMZ (Keyhole Markup Zip) files for Google Earth
- **Features**: NHC cone tracking, weather outlook integration, platform alerts
- **Dependencies**: 
  - Core: requests (HTTP client)
  - Optional: matplotlib (chart generation), shapely (geometry operations)

## Common Tasks

### Working with the Main Script
- **Script location**: Always use `scripts/cstar_kml.py`
- **Default output**: `Cstar_Locations/cstar_locations.kmz`
- **Configuration**: No config files - all options via command line arguments
- **Icons**: Script references `./icons/<size>/pc#.png` but no icons directory exists by default
- **Network timeout**: Script handles network failures gracefully, typically exits within 1 second when offline

### Making Changes
- **NO BUILD REQUIRED**: Python script runs directly after dependency installation
- **Testing approach**: Manual execution testing only - no automated test framework
- **Code style**: No enforced linting rules - follow existing code style in script
- **Dependencies**: Avoid adding new dependencies unless absolutely necessary
- **Output validation**: Generated KMZ files should be valid ZIP archives containing KML data

## Timing Expectations
- **Dependency installation**: 1-5 seconds for core requirements, 15+ seconds for optional packages
- **Script compilation check**: <1 second  
- **Script execution (offline)**: <1 second to fail gracefully
- **Script execution (online)**: Variable depending on data fetching - can take minutes
- **NEVER CANCEL**: Network operations may appear to hang - wait at least 60 seconds before considering alternatives

## Troubleshooting

### Common Issues
- **Import errors**: Install missing dependencies with pip3
- **Network timeouts**: Expected when ERDDAP servers are unreachable - script handles gracefully
- **Optional dependency failures**: matplotlib/shapely installation may timeout - not required for basic functionality
- **Permission errors**: Use `pip3 install --user` if system pip fails

### Expected Behavior
- Script runs without network access but fails at data fetching stage
- Missing optional dependencies result in warnings but script continues
- Generated KMZ files are standard ZIP archives viewable in Google Earth
- Script provides colored console output with status indicators (✓, ⚠️, ❌)

## Development Guidelines
- **Minimal changes**: Make surgical modifications to the single Python script
- **No new infrastructure**: Do not add build systems, test frameworks, or linting configs
- **Preserve functionality**: Maintain existing command-line interface and output format
- **Network resilience**: Ensure changes handle network failures gracefully
- **Documentation**: Update this file if significant functionality changes are made