#include <iostream>
class Node {
public:
    int data;
    Node* next;
    Node(int value) : data(value), next(nullptr){}
};
class SortedList {
private:
    Node* head;
    
public:
    SortedList() : head(nullptr){}
    
    // Insert and sort the entire linked list
    void insert(Node* node) {
        if(node == nullptr) return;
        
        // Store next node to be processed
        Node* next = node->next;
        
        // Insert current node into sorted position
        insertNode(node);
        
        // Recursively process remaining nodes
        insert(next);
    }
    
private:
    // Insert single node into appropriate position
    void insertNode(Node* node) {
        node->next = nullptr;  // Disconnect from original list
        
        if (head == nullptr || head->data >= node->data) {
            node->next = head;
            head = node;
            return;
        }
        
        Node* current = head;
        while (current->next != nullptr && current->next->data < node->data) {
            current = current->next;
        }
        
        node->next = current->next;
        current->next = node;
    }
    
public:
    void print() {
        Node* current = head;
        while (current != nullptr) {
            std::cout << current->data << " -> ";
            current = current->next;
        }
        std::cout << "nullptr" << std::endl;
    }
    
    ~SortedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }
};
int main() {
    // Create original list
    Node* head = nullptr;
    Node* current = nullptr;
    
    int values[] = {5, 3, 8, 1, 9};
    
    for(int value : values) {
        if(head == nullptr) {
            head = new Node(value);
            current = head;
        } else {
            current->next = new Node(value);
            current = current->next;
        }
    }
    
    // Print original list
    std::cout << "Original list: ";
    current = head;
    while(current != nullptr) {
        std::cout << current->data << " -> ";
        current = current->next;
    }
    std::cout << "nullptr" << std::endl;
    
    // Sort list
    SortedList sortedList;
    sortedList.insert(head);
    
    // Print sorted list
    std::cout << "Sorted list: ";
    sortedList.print();
    
    return 0;
}