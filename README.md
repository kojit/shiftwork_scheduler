# Schedule shift work
This script schedules shift work by using pulp, a linear problem solver.

In order to schedule, it is necessary first to prepare an Excel file or
a tab-delimited text filled out the conditions and indivisual requests.

Each shift type name must be a single character and defined as a string,
e.g. 'DNGO'(Day, Night, Graveyard, Off, etc), in Japanese '日夜明休'.

Also you may want to specify the shift patterns good and prohibitted.
