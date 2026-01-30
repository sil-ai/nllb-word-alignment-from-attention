# Test Data

This directory should contain aligned text files for testing.

## Data Format

The comparison script expects:
- **Source file**: One sentence per line (e.g., English Bible verses)
- **Target file**: One sentence per line (e.g., target language Bible verses)
- Lines must be **aligned** (line N in source corresponds to line N in target)

## Example Format

**eng-source.txt:**
```
In the beginning God created the heavens and the earth.
Now the earth was formless and empty, darkness was over the surface of the deep.
And the Spirit of God was hovering over the waters.
```

**nih-target.txt:**
```
Mwanzoni Mungu aliumba mbingu na dunia.
Dunia ilikuwa tupu na haikuwa na umbo.
Roho wa Mungu alikuwa akielea juu ya maji.
```

## Obtaining Bible Texts

For Bible translation work, aligned texts can be obtained from:

1. **Digital Bible Library (DBL)**: https://thedigitalbiblelibrary.org/
   - Licensed Bible texts for approved organizations

2. **eBible.org**: https://ebible.org/
   - Public domain and freely licensed translations

3. **Paratext**: Export from your Paratext project
   - Export as plain text, one verse per line

## VREF Format

The standard "vref" format uses 31,102 lines for the Protestant Bible canon, with each line corresponding to a specific verse reference. Empty lines indicate verses that don't exist in that translation.

To create vref-aligned files:
1. Export your translation with verse references
2. Map each verse to the canonical line number
3. Leave blank lines for missing verses

## Note

User-specific Bible text files are not included in this repository due to licensing restrictions. You must obtain your own aligned text files for testing.
