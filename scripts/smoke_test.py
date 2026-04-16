"""Quick smoke test for predict.py"""
import sys
sys.path.insert(0, ".")
import predict

a = predict.load_artifacts()
print("Loaded institutions:", list(a["institutions"].keys()))

cases = [
    ("ACC", "211", "PRINCIPLES OF ACCOUNTING I", "Introduces accounting principles."),
    ("", "", "Calculus I", "Limits, derivatives, integrals of single-variable functions."),
    ("CSC", "221", "Introduction to Programming", "Problem solving with Python."),
]

for dept, num, title, desc in cases:
    print(f"\n--- '{dept}' '{num}' '{title}' ---")
    r = predict.predict_transfer(dept, num, title, desc, institutions=["wm", "vt"], top_k=3)
    for inst, matches in r.items():
        print(f"  {inst}:")
        for m in matches:
            print(f"    {m['code']:<14} conf={m['confidence']:.3f} [{m['confidence_label']}]  {m['title']}")

print("\nSMOKE TEST PASSED")
