import zipfile
import os

xlsx_path = 'dataset/images_info.xlsx'
if os.path.exists(xlsx_path):
    with zipfile.ZipFile(xlsx_path, 'r') as z:
        print("Internal File List:")
        for name in z.namelist()[:20]: # Show first 20 internal files
            print(f"  {name}")
        
        # Check for media (images)
        media_files = [n for n in z.namelist() if 'media' in n.lower()]
        if media_files:
            print(f"\nFound {len(media_files)} media files inside Excel.")
            for mf in media_files[:5]:
                print(f"  {mf}")
        else:
            print("\nNo media files found inside Excel.")
            
        # Check for sheet data
        sheet_files = [n for n in z.namelist() if 'worksheets/sheet' in n.lower()]
        for sf in sheet_files:
            with z.open(sf) as f:
                content = f.read().decode('utf-8')
                # Very rough peek into XML to see if there are paths
                if '.jpg' in content.lower() or '.png' in content.lower():
                    print(f"  Sheet {sf} seems to reference image files.")
else:
    print(f"File {xlsx_path} not found.")
