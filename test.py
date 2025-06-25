import requests
from bs4 import BeautifulSoup

# I'm using beautifulsoup4 to parse the HTML content
# and get the data from the table elements.
# because for some reason, the url cannot be converted to a CSV file directly.
def decode(url):
    # Convert published link to HTML view
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print(response.text)

    coordinates = []

    # Find all rows in the given table (<tr> element)
    # remove the header row
    rows = soup.find_all('tr')[1:]  # remove the header

    # Start extracting the string character and coordinates from the table
    # Start with finding all <td> elements in each row
    # Strip the first (0th) and last (2nd) columns as coordinates
    # the middle column is the character
    for row in rows:
        cols = row.find_all('td')
        # doesn't hurt to be careful with the number of columns
        if len(cols) != 3:
            continue
        try:
            x = int(cols[0].text.strip())
            char = cols[1].text.strip()
            y = int(cols[2].text.strip())
            coordinates.append((x, char, y))
        except ValueError:
            continue

    if not coordinates:
        print("Oops, no valid values found.")
        return

    # get the maximum x and y values to determine the grid size
    max_x = max(x for x, _, _ in coordinates)
    max_y = max(y for _, _, y in coordinates)

    # initialize the grid matrix with space characters
    # with the size of (max_x + 1) x (max_y + 1)
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Place the characters in the grid based on coordinates
    for x, char, y in coordinates:
        grid[y][x] = char

    # Print the grid
    for row in grid:
        print(''.join(row))

decode("https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub")
