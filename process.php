<?php
header('Content-Type: application/json');

// Check if file was uploaded
if (empty($_FILES['files'])) {
    echo json_encode(['success' => false, 'error' => 'No files uploaded']);
    exit;
}

$file = $_FILES['files']['tmp_name'][0];
$filename = $_FILES['files']['name'][0];
$extension = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
$text = '';

try {
    // Process different file types
    switch ($extension) {
        case 'pdf':
            require 'vendor/autoload.php';
            $parser = new \Smalot\PdfParser\Parser();
            $pdf = $parser->parseFile($file);
            $text = $pdf->getText();
            break;
            
        case 'docx':
            require 'vendor/autoload.php';
            $phpWord = \PhpOffice\PhpWord\IOFactory::load($file);
            foreach ($phpWord->getSections() as $section) {
                foreach ($section->getElements() as $element) {
                    if ($element instanceof \PhpOffice\PhpWord\Element\TextRun) {
                        foreach ($element->getElements() as $textElement) {
                            if ($textElement instanceof \PhpOffice\PhpWord\Element\Text) {
                                $text .= $textElement->getText() . ' ';
                            }
                        }
                    }
                }
            }
            break;
            
        case 'pptx':
            require 'vendor/autoload.php';
            $presentation = \PhpOffice\PhpPresentation\IOFactory::load($file);
            foreach ($presentation->getSlides() as $slide) {
                foreach ($slide->getShapeCollection() as $shape) {
                    if ($shape instanceof \PhpOffice\PhpPresentation\Shape\RichText) {
                        foreach ($shape->getParagraphs() as $paragraph) {
                            foreach ($paragraph->getRichTextElements() as $element) {
                                $text .= $element->getText() . ' ';
                            }
                        }
                    }
                }
            }
            break;
            
        case 'txt':
            $text = file_get_contents($file);
            break;
            
        default:
            throw new Exception("Unsupported file type: $extension");
    }

    // Generate summary (simplified - in production use NLP libraries)
    $summary = generateSummary($text, $_POST['length']);
    
    echo json_encode([
        'success' => true,
        'summary' => $summary,
        'filename' => $filename
    ]);
    
} catch (Exception $e) {
    echo json_encode(['success' => false, 'error' => $e->getMessage()]);
}

function generateSummary($text, $length) {
    $sentences = preg_split('/(?<=[.?!])\s+/', $text, -1, PREG_SPLIT_NO_EMPTY);
    $summaryLength = max(1, (int)($length / 100 * count($sentences)));
    return implode(' ', array_slice($sentences, 0, $summaryLength));
}
?>