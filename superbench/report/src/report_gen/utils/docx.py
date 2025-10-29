from pdf2docx import Converter

def convert_pdf_to_word(pdf_file, word_file):
    # Create a Converter object
    cv = Converter(pdf_file)

    # Convert the PDF to a Word document
    cv.convert(word_file, start=0, end=None)

    # Close the Converter object
    cv.close()