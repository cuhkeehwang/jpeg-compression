import sys
from Compression import *
from GUI import *

app = QtGui.QApplication(sys.argv)
Form = QtGui.QWidget()
ui = Ui_Form()
ui.setupUi(Form)

Form.show()
sys.exit(app.exec_())