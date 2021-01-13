## How to Use

- Add raw data folder into this directory `DataCleaner/`
    - Raw data folder usually named like this: `yyyymmdd-<timestamp>`
    - It contains:
        - `images` folder
        - .h264 file
        - .json file
        - .mp4 file
- Create directory `mkdir cleaned_all` and `mkdir cleaned_all_masked` for output file directory
- Run `python clean_json_asRCSnail.py` or `python clean_json_asRCSnail_masked.py`

## Credits
`clean_json_asRCSnail.py` and `clean_json_asRCSnail_masked.py` from Ardi