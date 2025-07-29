import java.awt.event.*;
import javax.swing.*;

public class GradeCalculator extends JFrame implements ActionListener {
    JButton sgpaButton, cgpaButton, exitButton;

    public GradeCalculator() {
        setTitle("SGPA & CGPA Calculator");
        setSize(400, 300);
        setLayout(null);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLocationRelativeTo(null); 

        sgpaButton = new JButton("Calculate SGPA");
        sgpaButton.setBounds(50, 30, 300, 30);
        sgpaButton.addActionListener(this);
        add(sgpaButton);

        cgpaButton = new JButton("Calculate CGPA");
        cgpaButton.setBounds(50, 70, 300, 30);
        cgpaButton.addActionListener(this);
        add(cgpaButton);

       exitButton = new JButton("Exit");
       exitButton.setBounds(100, 110, 200, 30); 
       exitButton.addActionListener(this);
       add(exitButton);


        setVisible(true);
    }

    // Convert letter grade to grade point
    int getGradePoint(String grade) {
        switch (grade.toUpperCase()) {
            case "O": return 10;
            case "A+": return 9;
            case "A": return 8;
            case "B+": return 7;
            case "B": return 6;
            case "C+": return 5;
            case "C": return 4;
            case "F": return 0;
            default:
                JOptionPane.showMessageDialog(this, "Invalid grade entered. Defaulting to 0.");
                return 0;
        }
    }

    void calculateSGPA() {
        try {
            int subjects = Integer.parseInt(JOptionPane.showInputDialog(this, "Enter number of subjects:"));
            float totalCredits = 0, weightedPoints = 0;

            for (int i = 1; i <= subjects; i++) {
                float credits = Float.parseFloat(JOptionPane.showInputDialog(this, "Subject " + i + " - Enter credits:"));
                String grade = JOptionPane.showInputDialog(this, "Subject " + i + " - Enter grade (O, A+, A, B+, B, C+, C, F):");
                int gradePoint = getGradePoint(grade);

                totalCredits += credits;
                weightedPoints += (credits * gradePoint);
            }

            float sgpa = weightedPoints / totalCredits;
            JOptionPane.showMessageDialog(this, String.format("SGPA for this semester = %.2f", sgpa));

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Invalid input! Please try again.");
        }
    }

    void calculateCGPA() {
        try {
            int semesters = Integer.parseInt(JOptionPane.showInputDialog(this, "Enter number of semesters:"));
            float totalSGPA = 0;

            for (int i = 1; i <= semesters; i++) {
                float sgpa = Float.parseFloat(JOptionPane.showInputDialog(this, "Enter SGPA for semester " + i + ":"));
                totalSGPA += sgpa;
            }

            float cgpa = totalSGPA / semesters;
            JOptionPane.showMessageDialog(this, String.format("CGPA = %.2f", cgpa));

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Invalid input! Please try again.");
        }
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == sgpaButton) {
            calculateSGPA();
        } else if (e.getSource() == cgpaButton) {
            calculateCGPA();
        } else if (e.getSource() == exitButton) {
            System.exit(0);
        }
    }

    public static void main(String[] args) {
        new GradeCalculator();
    }
}
