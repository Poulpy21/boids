
#ifndef TRISTATE_H
#define TRISTATE_H
        
enum State {
    LOW     = 0,
    MIDDLE  = 1,
    HIGH    = 2
};

struct TriState {

    public:
        TriState() : val(0) {} 
        TriState(char val) : val(val) {} 
        TriState(const TriState &other) : val(other.getVal()) {} 

        int getVal() const { return val; }
        State getState() const { return static_cast<State>(val); };

        operator unsigned int() const { return val; }

    private:
        char val;
};

#endif /* end of include guard: TRISTATE_H */
